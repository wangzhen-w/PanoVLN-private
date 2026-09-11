"""CLI for on-demand rendering, local generation/repair and strict R2R export."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import shutil
import tempfile
import time
import traceback

from tqdm.auto import tqdm

from dataset_create.instruction import SCHEMA_VERSION
from dataset_create.instruction.client import QwenClient, digest
from dataset_create.instruction.export import make_r2r_episode, validate_r2r, write_r2r
from dataset_create.instruction.language import LanguageContractError, generate_local, polish_and_assemble
from dataset_create.instruction.rendering import EpisodeRenderer
from dataset_create.instruction.segmentation import EvidenceError, Segment, merge_for_repair, segment_episode
from dataset_create.instruction.verification import repair_feedback, verify_segment
from dataset_create.instruction.training import export_clean_erp, validate_erp
from dataset_create.trajectory.hm3d import resolve_scene_path, shortest_path
from dataset_create.trajectory.io_utils import atomic_json_dump, load_json, load_json_gz
from dataset_create.trajectory.schema import validate_dataset


DEFAULT_CONFIG = Path(__file__).parent / "config/default.json"


def implementation_digest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(list((root / "instruction").glob("*.py")) + list((root / "trajectory").glob("*.py")))
    return digest([(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])


def episode_key(episode):
    # No user-controlled trajectory id can escape its workspace directory.
    return hashlib.sha256(episode["trajectory_id"].encode()).hexdigest()[:24]


def episode_signature(episode, manifest):
    scene = resolve_scene_path(manifest["scene_root"], episode["scene_id"])
    asset_stats = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in (scene, scene.with_suffix(".navmesh"))]
    return digest([episode, manifest["behavior_fingerprint"], asset_stats])


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another run owns {path}; use a separate run name") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _media_available(evidence):
    paths = [evidence["marked_video"], evidence["clean_video"], *evidence["clean_frames"], *evidence["marked_frames"]]
    for decision in evidence["decisions"]:
        paths.extend([decision["marked_compass"], decision["clean_compass"]])
    if evidence["stop"]:
        paths.append(evidence["stop"]["actual"]["image"])
    if evidence.get("clean_storyboard"):
        paths.append(evidence["clean_storyboard"])
    return all(Path(p).is_file() for p in paths)


def validate_acceptance(record):
    checks = record.get("verification", [])
    if not checks or not all(r["passed"] for r in checks):
        raise ValueError("Accepted record lacks successful final verification")
    decisions = [r.get("decision_id") for r in checks if r["kind"] == "decision"]
    expected = [f"d{i}" for i in range(len(record["source_trajectory"]["decision_events"]))]
    if sorted(decisions) != sorted(expected) or len(decisions) != len(expected):
        raise ValueError("Accepted record is missing or duplicating decision checks")
    segment_ids = [s["segment_id"] for s in record["segments"]]
    if sorted(r["segment_id"] for r in checks if r["kind"] == "motion") != sorted(segment_ids):
        raise ValueError("Accepted record is missing a segment motion check")
    stops = [r for r in checks if r["kind"] == "stop"]
    if len(stops) != 1 or stops[0]["segment_id"] != segment_ids[-1]:
        raise ValueError("Accepted record lacks the terminal stop check")
    for check in checks:
        if check["response"].get("status") != "pass":
            raise ValueError("Accepted record has a failed local route, arrival or movement review")
        if check["kind"] == "decision" and (
                check["observation"].get("uncertain") is not False or
                check["response"].get("path_matches") is not True or
                check["response"].get("choice_is_clear") is not True):
            raise ValueError("Accepted decision lacks reliable, consistent and unambiguous route evidence")
    text = record["final"]["instruction_text"]
    if record.get("acceptance", {}).get("final_instruction_hash") != digest(text):
        raise ValueError("Accepted text hash differs from verified instruction")
    if record["r2r_episode"]["instruction"]["instruction_text"] != text:
        raise ValueError("R2R text diverges from verified final instruction")


def process_episode(renderer, client, episode, index, manifest, settings, mode, retry_quarantined=False):
    root = Path(manifest["work_dir"]) / "episodes" / episode_key(episode)
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "record.json"
    signature = episode_signature(episode, manifest)
    record = load_json(record_path) if record_path.is_file() else {}
    if record and record.get("fingerprint") != signature:
        raise RuntimeError(f"Stale episode cache {record_path}; use a new --name or --work-dir")
    if record.get("status") == "accepted" or (record.get("status") == "quarantined" and not retry_quarantined):
        return {"status": record["status"], "resumed": True, "trajectory_id": episode["trajectory_id"]}
    if record.get("status") == "quarantined" and retry_quarantined:
        record["retry_epoch"] = record.get("retry_epoch", 0) + 1
        sid = record.get("failure", {}).get("segment_id")
        if sid is not None:
            record.get("locals", {}).pop(sid, None)
        else:
            record["locals"] = {}
        record["local_repairs"] = {}
        record["upstream_repairs"] = {}
    if mode == "render" and record.get("status") == "prepared" and all(_media_available(e) for e in record["evidence"].values()):
        return {"status": "prepared", "resumed": True, "trajectory_id": episode["trajectory_id"]}
    started = time.monotonic()
    record.update({"schema_version": SCHEMA_VERSION, "fingerprint": signature,
                   "generation_fingerprint": manifest["behavior_fingerprint"],
                   "trajectory_id": episode["trajectory_id"], "episode_index": index,
                   "source_trajectory": episode, "status": "working"})
    record.setdefault("history", [])
    record.setdefault("locals", {})
    record.setdefault("evidence", {})
    record.setdefault("local_repairs", {})
    record.setdefault("upstream_repairs", {})
    def checkpoint():
        record["elapsed_seconds"] = time.monotonic() - started + record.get("previous_elapsed_seconds", 0.)
        atomic_json_dump(record, record_path)
    if record.get("elapsed_seconds"):
        record["previous_elapsed_seconds"] = record["elapsed_seconds"]
    try:
        states = renderer.replay(episode)
        segments = [Segment(**s) for s in record["segments"]] if record.get("segments") else segment_episode(episode, states, settings["segmentation"])
        record["states"] = states
        record["segments"] = [s.to_dict() for s in segments]
        checkpoint()
        route = None
        def render(segment, revision=0):
            nonlocal route
            if route is None:
                route = renderer.ground_route(states)
                import numpy as np
                np.savez_compressed(root / "ground_route.npz", points=route.points, tangents=route.tangents,
                                    arc=route.arc, valid=route.valid)
                record["grounded_fraction"] = float(route.valid.mean())
            value = renderer.render_segment(episode, states, segment, route, root / "media",
                                            settings["segmentation"], revision)
            record["evidence"][segment.segment_id] = value
            record["segments"] = [s.to_dict() for s in segments]
            checkpoint()
            return value
        for segment in segments:
            evidence = record["evidence"].get(segment.segment_id)
            if evidence is None or not _media_available(evidence):
                render(segment, record["upstream_repairs"].get(segment.segment_id, 0))
        if mode == "render":
            record["status"] = "prepared"
            checkpoint()
            return {"status": "prepared", "trajectory_id": episode["trajectory_id"]}
        # Iterations are bounded per segment. API failures remain resumable errors;
        # ungroundable/ambiguous samples are explicitly quarantined.
        for _ in range((len(segments)+1) * (settings["validation"]["max_local_repairs"] + settings["validation"]["max_upstream_repairs"] + 2)):
            uncertain = None
            for i, segment in enumerate(segments):
                sid = segment.segment_id
                if sid not in record["locals"]:
                    previous = record["locals"][segments[i-1].segment_id]["text"] if i else ""
                    local = generate_local(client, segment, record["evidence"][sid], previous,
                                           record.get("feedback", {}).get(sid),
                                           record.get("retry_epoch", 0)*100 + record["local_repairs"].get(sid, 0))
                    record["locals"][sid] = local
                    checkpoint()
                local = record["locals"][sid]
                if local["uncertain"]:
                    uncertain = (segment, [{"kind": "author_uncertainty", "issue_type": local["issue_type"],
                                            "reason": str(local["issues"])}])
                    break
            if uncertain:
                failing_segment, feedback = uncertain
            else:
                final = polish_and_assemble(client, segments, record["locals"])
                record["final"] = final
                checkpoint()
                checks = []
                for segment in segments:
                    segment_checks = verify_segment(client, final, segment, record["evidence"][segment.segment_id], settings["validation"])
                    checks.extend(segment_checks)
                    record["verification"] = checks
                    checkpoint()
                    if any(not check["passed"] for check in segment_checks):
                        # Repair the earliest failing segment before paying for
                        # later checks. Acceptance still requires every segment.
                        break
                failures = [r for r in checks if not r["passed"]]
                if not failures:
                    final_path = shortest_path(renderer.sim.pathfinder, states[0]["position"], states[-1]["position"])
                    if final_path is None:
                        raise EvidenceError("no_geodesic_path_to_real_stop")
                    record["r2r_episode"] = make_r2r_episode(
                        episode, index, states, final["instruction_text"], final_path["distance"],
                        resolve_scene_path(manifest["scene_root"], episode["scene_id"]), manifest["scene_root"])
                    record["training_erp"] = export_clean_erp(renderer, episode, states, manifest["erp_root"], settings["erp"])
                    record["status"] = "accepted"
                    record["acceptance"] = {"all_decisions_verified": True, "real_stop_verified": True,
                                             "motion_verified": True, "final_instruction_hash": digest(final["instruction_text"])}
                    checkpoint()
                    return {"status": "accepted", "trajectory_id": episode["trajectory_id"]}
                sid = failures[0]["segment_id"]
                failing_segment = next(s for s in segments if s.segment_id == sid)
                feedback = repair_feedback([f for f in failures if f["segment_id"] == sid])
            sid = failing_segment.segment_id
            # The author also sees the marked route and can diagnose whether
            # wording or upstream material caused the clean review's difficulty.
            # Let it review the failed local text first; genuine material issues
            # return as author uncertainty and are repaired in stages 1/2.
            issue_types = {f["issue_type"] for f in feedback} if uncertain else {"language"}
            record["history"].append({"segment_id": sid, "feedback": feedback,
                                       "previous_local": record["locals"].get(sid),
                                       "previous_final": record.get("final"),
                                       "previous_verification": record.get("verification")})
            if issue_types & {"visual", "segmentation"}:
                count = record["upstream_repairs"].get(sid, 0)
                if count >= settings["validation"]["max_upstream_repairs"]:
                    raise EvidenceError(f"upstream_repair_exhausted:{sid}:{json.dumps(feedback)}")
                if "segmentation" in issue_types:
                    old_ids = {s.segment_id for s in segments}
                    segments, failing_segment = merge_for_repair(segments, sid)
                    new_ids = {s.segment_id for s in segments}
                    for removed in old_ids - new_ids:
                        record["locals"].pop(removed, None)
                    sid = failing_segment.segment_id
                record["upstream_repairs"][sid] = count + 1
                render(failing_segment, count + 1)
            else:
                count = record["local_repairs"].get(sid, 0)
                if count >= settings["validation"]["max_local_repairs"]:
                    record["status"] = "quarantined"
                    record["failure"] = {"kind": "local_verification_failed", "segment_id": sid, "feedback": feedback}
                    checkpoint()
                    return {"status": "quarantined", "trajectory_id": episode["trajectory_id"]}
                record["local_repairs"][sid] = count + 1
            record.setdefault("feedback", {})[sid] = {
                "previous_local_instruction": record["locals"].get(sid, {}).get("text"),
                "issues": feedback,
            }
            record["locals"].pop(sid, None)
            checkpoint()
        raise EvidenceError("repair_iteration_budget_exhausted")
    except (EvidenceError, LanguageContractError) as error:
        record["status"] = "quarantined"
        record["failure"] = {"kind": "upstream_evidence" if isinstance(error, EvidenceError) else "model_contract", "reason": str(error)}
    except Exception as error:
        record["status"] = "error"
        record["failure"] = {"kind": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()}
    checkpoint()
    return {"status": record["status"], "trajectory_id": episode["trajectory_id"], "failure": record.get("failure")}


def clear_episode_materials(root):
    """Release finished inference inputs; compact records remain until export."""
    for name in ("media", "requests"):
        shutil.rmtree(Path(root) / name, ignore_errors=True)
    (Path(root) / "ground_route.npz").unlink(missing_ok=True)


def clear_completed_work(work):
    """Keep the completion receipt until the last deletion, including on resume."""
    work = Path(work)
    for path in work.iterdir():
        if path.name == "manifest.json":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    (work / "manifest.json").unlink()
    work.rmdir()


def check_completed_output(output, name, items, manifest):
    existing = load_json_gz(output / f"{name}.json.gz")
    validate_r2r(existing, manifest["scene_root"])
    plain = output / f"{name}.json"
    if not plain.is_file() or load_json(plain) != existing:
        raise ValueError("Existing JSON/gzip outputs differ; resume with the intact run workspace")
    selected = {e["trajectory_id"]: e for _, e in items}
    for episode in existing["episodes"]:
        source = selected.get(episode["trajectory_id"])
        if source is None:
            raise ValueError("Output belongs to another selection; use a new output name")
        validate_erp(manifest["erp_root"], episode["trajectory_id"], len(source["action_ids"]), manifest["settings"]["erp"])
    return {"status": "already_exported", "accepted": len(existing["episodes"]),
            "dataset_path": str(output / f"{name}.json.gz"),
            "note": "Use a new output name for another selection or configuration."}


def run_worker(items, gpu, manifest, settings, mode, retry_quarantined, keep_work, report_progress):
    renderer = EpisodeRenderer(manifest["scene_root"], manifest["trajectory_metadata"], settings["render"], gpu, settings["erp"])
    client = None
    results = []
    try:
        for index, episode in items:
            root = Path(manifest["work_dir"]) / "episodes" / episode_key(episode)
            if mode == "generate":
                if client is None:
                    client = QwenClient(settings["model"], root / "requests")
                else:
                    client.cache_dir = root / "requests"
                    client.cache_dir.mkdir(parents=True, exist_ok=True)
            result = process_episode(renderer, client, episode, index, manifest, settings, mode, retry_quarantined)
            results.append(result)
            if not keep_work and result["status"] in {"accepted", "quarantined"}:
                clear_episode_materials(root)
            report_progress(result)
    finally:
        renderer.close()
    return {"pid": os.getpid(), "gpu_device_id": gpu, "results": results, "api_calls": client.calls if client else 0,
            "cache_hits": client.cache_hits if client else 0, "usage": client.usage if client else {}}


def run_workers(shards, worker_gpus, manifest, settings, mode, retry_quarantined, keep_work):
    counts = Counter()
    fields = ["accepted", "quarantined", "error", "resumed"]
    if mode == "render":
        fields.insert(0, "prepared")
    with tqdm(total=sum(map(len, shards)), desc="Instructions" if mode == "generate" else "Rendering",
              unit="traj", dynamic_ncols=True, mininterval=1., postfix={key: 0 for key in fields}) as progress:
        def report(result):
            counts[result["status"]] += 1
            counts["resumed"] += bool(result.get("resumed"))
            progress.set_postfix({key: counts[key] for key in fields}, refresh=False)
            progress.update(1)
            if result["status"] == "error":
                progress.write(f"{result['trajectory_id']}: {str(result.get('failure', {}))[:800]}")

        if len(shards) == 1:
            return [run_worker(shards[0], worker_gpus[0], manifest, settings, mode,
                               retry_quarantined, keep_work, report)]
        context = multiprocessing.get_context("spawn")
        results = []
        with context.Manager() as manager:
            updates = manager.Queue()
            with ProcessPoolExecutor(max_workers=len(shards), mp_context=context) as pool:
                pending = {pool.submit(run_worker, shard, gpu, manifest, settings, mode,
                                       retry_quarantined, keep_work, updates.put)
                           for gpu, shard in zip(worker_gpus, shards) if shard}
                while pending:
                    try:
                        report(updates.get(timeout=.2))
                    except Empty:
                        pass
                    finished = {future for future in pending if future.done()}
                    for future in finished:
                        results.append(future.result())
                    pending -= finished
                # A worker can finish with its last per-trajectory reports still queued.
                while True:
                    try:
                        report(updates.get_nowait())
                    except Empty:
                        break
        return results


def select_episodes(dataset, args):
    items = []
    for index, episode in enumerate(dataset["episodes"]):
        if args.scene_ids and not set(episode["scene_id"].split("/")) & set(args.scene_ids):
            continue
        if args.trajectory_ids and episode["trajectory_id"] not in args.trajectory_ids:
            continue
        items.append((index, episode))
    if args.limit and args.selection == "diverse":
        groups = defaultdict(list)
        for item in items:
            groups[item[1]["scene_id"]].append(item)
        # Round-robin scenes, prioritize richer decisions within each scene.
        for group in groups.values():
            group.sort(key=lambda item: (-len(item[1]["decision_events"]), item[0]))
        selected = []
        depth = 0
        while len(selected) < min(args.limit, len(items)):
            for scene in sorted(groups):
                if depth < len(groups[scene]):
                    selected.append(groups[scene][depth])
                    if len(selected) == args.limit:
                        break
            depth += 1
        items = selected
    elif args.limit:
        items = items[:args.limit]
    return sorted(items, key=lambda item: (item[1]["scene_id"], item[0]))


def aggregate(items, manifest, output, name, allow_incomplete=False):
    accepted, counts, missing, reasons = [], Counter(), [], Counter()
    first_pass = 0
    erp_frames = 0
    for index, episode in items:
        path = Path(manifest["work_dir"]) / "episodes" / episode_key(episode) / "record.json"
        if not path.is_file():
            missing.append(episode["trajectory_id"])
            continue
        record = load_json(path)
        if record.get("fingerprint") != episode_signature(episode, manifest):
            raise RuntimeError(f"Incompatible episode record: {path}")
        counts[record["status"]] += 1
        if record["status"] == "accepted":
            validate_acceptance(record)
            count = len(record["source_trajectory"]["action_ids"])
            validate_erp(manifest["erp_root"], episode["trajectory_id"], count, manifest["settings"]["erp"])
            accepted.append(record["r2r_episode"])
            erp_frames += count
            first_pass += not any(record.get("local_repairs", {}).values()) and not any(record.get("upstream_repairs", {}).values())
        else:
            failure = record.get("failure", {})
            reasons[failure.get("reason", failure.get("kind", record["status"])).split(":")[0]] += 1
    incomplete = len(missing) + sum(counts[s] for s in ("working", "error", "prepared"))
    summary = {"schema_version": SCHEMA_VERSION, "selected": len(items), "statuses": dict(counts),
               "incomplete": incomplete, "missing": missing, "failure_reasons": dict(reasons),
               "accepted_fraction": len(accepted)/max(1, len(items)),
               "first_pass_accepted": first_pass, "accepted_after_repair": len(accepted) - first_pass,
               "erp_frames": erp_frames, "erp_root": manifest["erp_root"],
               "json_path": str(output / f"{name}.json"),
               "dataset_path": str(output / f"{name}.json.gz"), "work_dir": manifest["work_dir"],
               "scene_root": manifest["scene_root"], "behavior_fingerprint": manifest["behavior_fingerprint"]}
    run_path = Path(manifest["work_dir"]) / "last_run.json"
    if run_path.is_file():
        workers = load_json(run_path)
        summary["last_run"] = {"processes": len(workers),
                               "api_calls": sum(w["api_calls"] for w in workers),
                               "cache_hits": sum(w["cache_hits"] for w in workers),
                               "usage": dict(sum((Counter(w["usage"]) for w in workers), Counter()))}
    atomic_json_dump(summary, Path(manifest["work_dir"]) / "summary.json")
    if incomplete and not allow_incomplete:
        raise RuntimeError(f"{incomplete} episodes incomplete; resume first, or explicitly export with --allow-incomplete")
    write_r2r(accepted, output / f"{name}.json.gz", manifest["scene_root"])
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--trajectories", required=True, type=Path)
    common.add_argument("--scene-root", type=Path, default=Path("/workspace/data2/dataset/general_VLN_data/HM3D"))
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--name", default="instructions")
    common.add_argument("--work-dir", type=Path)
    common.add_argument("--erp-root", type=Path)
    common.add_argument("--erp-width", type=int)
    common.add_argument("--erp-height", type=int)
    common.add_argument("--jpeg-quality", type=int)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--limit", type=int, default=0, help="0 means all filtered trajectories")
    common.add_argument("--selection", choices=["first", "diverse"], default="first")
    common.add_argument("--scene-ids", nargs="+", default=[])
    common.add_argument("--trajectory-ids", nargs="+", default=[])
    common.add_argument("--gpu-device-ids", nargs="+", type=int, default=[0])
    common.add_argument("--processes", type=int, default=1, help="Worker processes, assigned round-robin to the listed GPUs")
    common.add_argument("--base-url")
    common.add_argument("--model")
    common.add_argument("--media-mode", choices=["video", "frames"])
    common.add_argument("--retry-quarantined", action="store_true")
    common.add_argument("--keep-work", action="store_true", help="Keep intermediate materials for debugging; default cleans them after completion")
    common.add_argument("--allow-incomplete", action="store_true")
    for mode in ("inspect", "render", "generate", "export"):
        commands.add_parser(mode, parents=[common])
    check = commands.add_parser("validate")
    check.add_argument("--dataset", required=True, type=Path)
    check.add_argument("--scene-root", required=True, type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "validate":
        dataset = load_json_gz(args.dataset) if args.dataset.suffix == ".gz" else load_json(args.dataset)
        validate_r2r(dataset, args.scene_root)
        print(json.dumps({"valid": True, "episodes": len(dataset["episodes"]), "dataset": str(args.dataset)}))
        return 0
    if args.limit < 0 or args.processes < 1 or len(set(args.gpu_device_ids)) != len(args.gpu_device_ids):
        raise ValueError("limit must be nonnegative; processes positive; GPU IDs unique")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("name must be a single filename component")
    dataset = load_json_gz(args.trajectories) if args.trajectories.suffix == ".gz" else load_json(args.trajectories)
    validate_dataset(dataset)
    items = select_episodes(dataset, args)
    if not items:
        raise ValueError("No trajectories selected")
    settings = load_json(args.config)
    for value, key in [(args.erp_width, "width"), (args.erp_height, "height"), (args.jpeg_quality, "jpeg_quality")]:
        if value is not None:
            settings["erp"][key] = value
    erp = settings["erp"]
    if erp["height"] <= 0 or erp["width"] != 2 * erp["height"] or not 1 <= erp["jpeg_quality"] <= 100:
        raise ValueError("ERP must have a positive 2:1 resolution and JPEG quality in [1, 100]")
    for option, key in [(args.base_url, "base_url"), (args.model, "name"), (args.media_mode, "media_mode")]:
        if option is not None:
            settings["model"][key] = option
    if args.mode == "inspect":
        print(json.dumps({"total_episodes": len(dataset["episodes"]), "selected": len(items),
                          "scenes": len({e["scene_id"] for _, e in items}),
                          "decision_counts": dict(Counter(len(e["decision_events"]) for _, e in items)),
                          "processes": args.processes, "gpu_device_ids": args.gpu_device_ids,
                          "metadata": dataset["metadata"], "settings": settings}, indent=2))
        return 0
    output = args.output_root.resolve()
    if args.trajectories.resolve() in {output / f"{args.name}.json.gz", output / f"{args.name}.json"}:
        raise ValueError("Output must never overwrite the source trajectories")
    work = (args.work_dir or output / ".work" / "instruction").resolve()
    erp_root = (args.erp_root or output / "image").resolve()
    if work == output or work in output.parents or work == erp_root or work in erp_root.parents:
        raise ValueError("Work directory must be separate from final output and ERP directories")
    if work == args.trajectories.parent.resolve() or work == args.scene_root.resolve():
        raise ValueError("Work directory must not be an input directory")
    manifest = {"schema_version": SCHEMA_VERSION, "source": str(args.trajectories.resolve()),
                "output_root": str(output), "name": args.name,
                "scene_root": str(args.scene_root.resolve()), "trajectory_metadata": dataset["metadata"],
                "settings": settings, "implementation_digest": implementation_digest(), "work_dir": str(work),
                "erp_root": str(erp_root)}
    manifest["behavior_fingerprint"] = digest(manifest)
    manifest["selected_trajectory_ids"] = [e["trajectory_id"] for _, e in items]
    lock = Path(tempfile.gettempdir()) / "panovln-locks" / (digest([str(output), args.name]) + ".lock")
    with run_lock(lock):
        manifest_path = work / "manifest.json"
        if not manifest_path.is_file() and work.exists() and any(work.iterdir()):
            raise ValueError(f"Work directory is not an owned pipeline workspace: {work}")
        if not manifest_path.is_file() and (output / f"{args.name}.json.gz").is_file():
            print(json.dumps(check_completed_output(output, args.name, items, manifest)))
            if work.is_dir():
                work.rmdir()
            return 0
        if manifest_path.is_file():
            previous = load_json(manifest_path)
            if previous["behavior_fingerprint"] != manifest["behavior_fingerprint"]:
                raise RuntimeError(f"Code, config, source or scene root changed since {manifest_path}. Use a new --name/--work-dir.")
            if previous.get("export_complete"):
                if previous["selected_trajectory_ids"] != manifest["selected_trajectory_ids"]:
                    raise ValueError("This run completed; use a new output name for another selection")
                print(json.dumps(check_completed_output(output, args.name, items, manifest)))
                if not args.keep_work:
                    clear_completed_work(work)
                return 0
        atomic_json_dump(manifest, manifest_path)
        if args.mode == "export":
            summary = aggregate(items, manifest, output, args.name, args.allow_incomplete)
        else:
            if args.mode == "generate":
                QwenClient(settings["model"], work / "requests").healthcheck()
            groups = defaultdict(list)
            for item in items:
                groups[item[1]["scene_id"]].append(item)
            worker_gpus = [args.gpu_device_ids[i % len(args.gpu_device_ids)] for i in range(args.processes)]
            shards = [[] for _ in worker_gpus]
            for group in sorted(groups.values(), key=len, reverse=True):
                min(shards, key=len).extend(group)
            worker_results = run_workers(shards, worker_gpus, manifest, settings, args.mode,
                                         args.retry_quarantined, args.keep_work)
            atomic_json_dump(worker_results, work / "last_run.json")
            if args.mode == "render":
                counts = Counter(r["status"] for w in worker_results for r in w["results"])
                print(json.dumps({"mode": "render", "statuses": dict(counts), "work_dir": str(work)}))
                return 1 if counts["error"] else 0
            summary = aggregate(items, manifest, output, args.name, args.allow_incomplete)
        print(json.dumps(summary, indent=2))
        if not args.keep_work and not summary["incomplete"]:
            # If cleanup itself is interrupted, resume validates the completed
            # outputs and finishes deletion without running generation again.
            atomic_json_dump({**manifest, "export_complete": True}, manifest_path)
            clear_completed_work(work)
        return 0 if summary["statuses"].get("accepted", 0) and not summary["incomplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import time
import traceback

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.habitat_shortest_path import (
    DEFAULT_GOAL_RADIUS,
    CONFIG,
    ShortestPathRolloutError,
    build_locality_balanced_episode_splits,
    build_worker_assignments,
    default_output_path,
    extract_instruction,
    filter_episodes,
    group_episodes_by_scene,
    habitat,
    load_dataset,
    parse_episode_ids,
    parse_gpu_ids,
    remap_gpu_ids_to_visible_devices,
    rollout_shortest_path_episode,
    sort_episodes_for_scene_locality,
    silence_external_output,
    validate_gpu_ids,
)

QUEUE_POLL_TIMEOUT_SECONDS = 5


def write_jsonl(output_path, items):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"saved {len(items)} samples to {output_path}")


def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def annotation_sort_key(item):
    return int(item["episode_id"])


def append_jsonl_item(handle, item):
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()


def skipped_output_path(output_path):
    root, ext = os.path.splitext(output_path)
    if ext == ".jsonl":
        return f"{root}.skipped.jsonl"
    return output_path + ".skipped.jsonl"


def remove_if_exists(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def load_annotation_index(path, tolerate_partial=False):
    annotation_index = {}
    if not os.path.exists(path):
        return annotation_index

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                if tolerate_partial:
                    print(
                        f"warning: ignoring malformed trailing line in {path} "
                        f"at line {line_number}"
                    )
                    break
                raise

            annotation_index[int(item["episode_id"])] = item

    return annotation_index


def write_sorted_annotation_index(output_path, annotation_index):
    items = sorted(annotation_index.values(), key=annotation_sort_key)
    write_jsonl(output_path, items)


def progress_dir_path(output_path, temp_root=None):
    if temp_root is not None:
        stable_name = hashlib.sha1(output_path.encode("utf-8")).hexdigest()[:16]
        return os.path.join(
            temp_root,
            "preprocess_inprogress",
            f"{stable_name}_{os.path.basename(output_path)}.inprogress",
        )
    return output_path + ".inprogress"


def rank_output_path(progress_dir, worker_index):
    return os.path.join(progress_dir, f"rank_{worker_index:02d}.jsonl")


def rank_skipped_output_path(progress_dir, worker_index):
    return os.path.join(progress_dir, f"rank_{worker_index:02d}.skipped.jsonl")


def list_rank_output_paths(progress_dir):
    if not os.path.isdir(progress_dir):
        return []

    return sorted(
        os.path.join(progress_dir, file_name)
        for file_name in os.listdir(progress_dir)
        if file_name.startswith("rank_") and file_name.endswith(".jsonl")
        and not file_name.endswith(".skipped.jsonl")
    )


def list_rank_skipped_output_paths(progress_dir):
    if not os.path.isdir(progress_dir):
        return []

    return sorted(
        os.path.join(progress_dir, file_name)
        for file_name in os.listdir(progress_dir)
        if file_name.startswith("rank_") and file_name.endswith(".skipped.jsonl")
    )


def load_annotation_index_from_paths(paths, tolerate_partial=False):
    annotation_index = {}
    for path in paths:
        annotation_index.update(
            load_annotation_index(path, tolerate_partial=tolerate_partial)
        )
    return annotation_index


def make_skipped_item(dataset_name, episode, error):
    return {
        "episode_id": int(episode.episode_id),
        "dataset": dataset_name,
        "scene_id": getattr(episode, "scene_id", ""),
        "instruction": extract_instruction(episode),
        "reason": "shortest_path_rollout_failed",
        "error": str(error),
    }


def configure_action_only_env(env_config, local_gpu_id=None):
    with habitat.config.read_write(env_config):
        if local_gpu_id is not None:
            env_config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_gpu_id

        rgb_sensor = env_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        rgb_sensor.width = 1
        rgb_sensor.height = 1

        env_config.habitat.task.measurements = {}


def finalize_annotation_outputs(output_path, progress_dir, merge_existing_output, cleanup=True):
    final_annotation_index = {}
    if merge_existing_output:
        final_annotation_index.update(load_annotation_index(output_path))
    final_annotation_index.update(
        load_annotation_index_from_paths(
            list_rank_output_paths(progress_dir),
            tolerate_partial=True,
        )
    )
    write_sorted_annotation_index(output_path, final_annotation_index)
    if cleanup:
        remove_if_exists(progress_dir)


def finalize_skipped_outputs(skipped_path, progress_dir, merge_existing_output, cleanup=True):
    final_skipped_index = {}
    if merge_existing_output:
        final_skipped_index.update(load_annotation_index(skipped_path))
    final_skipped_index.update(
        load_annotation_index_from_paths(
            list_rank_skipped_output_paths(progress_dir),
            tolerate_partial=True,
        )
    )
    write_sorted_annotation_index(skipped_path, final_skipped_index)
    if cleanup:
        remove_if_exists(progress_dir)


def finalize_preprocess_outputs(output_path, progress_dir, merge_existing_output):
    finalize_annotation_outputs(
        output_path=output_path,
        progress_dir=progress_dir,
        merge_existing_output=merge_existing_output,
        cleanup=False,
    )
    finalize_skipped_outputs(
        skipped_path=skipped_output_path(output_path),
        progress_dir=progress_dir,
        merge_existing_output=merge_existing_output,
        cleanup=True,
    )


def build_worker_plans(selected_episodes, worker_assignments):
    episode_splits = build_locality_balanced_episode_splits(
        selected_episodes,
        len(worker_assignments),
    )
    worker_plans = []
    for assignment, chunk_episodes in zip(worker_assignments, episode_splits):
        if not chunk_episodes:
            continue

        worker_plans.append(
            {
                "assignment": assignment,
                "scene_ids": sorted({episode.scene_id for episode in chunk_episodes}),
                "episode_ids": [int(episode.episode_id) for episode in chunk_episodes],
                "episode_count": len(chunk_episodes),
            }
        )

    return worker_plans


def generate_annotations_sequential(
    env_config,
    dataset,
    selected_episodes,
    dataset_name,
    goal_radius,
    progress_output_path,
    skipped_progress_output_path,
    total_selected_count,
    num_skipped,
):
    dataset.episodes = sort_episodes_for_scene_locality(selected_episodes)
    configure_action_only_env(env_config)

    env = None
    try:
        with silence_external_output():
            env = habitat.Env(config=env_config.habitat, dataset=dataset)

        progress = tqdm(
            total=total_selected_count,
            desc=f"preprocess:{dataset_name}",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        if num_skipped:
            progress.update(num_skipped)
            progress.set_postfix_str(
                f"done={num_skipped}/{total_selected_count}, skipped={num_skipped}"
            )

        processed_count = num_skipped
        dropped_count = 0
        os.makedirs(os.path.dirname(progress_output_path), exist_ok=True)
        with open(progress_output_path, "a", encoding="utf-8") as progress_handle, open(
            skipped_progress_output_path, "a", encoding="utf-8"
        ) as skipped_handle:
            for episode in dataset.episodes:
                episode_start_time = time.time()
                try:
                    item = rollout_shortest_path_episode(
                        env=env,
                        episode=episode,
                        goal_radius=goal_radius,
                    )
                except ShortestPathRolloutError as error:
                    append_jsonl_item(
                        skipped_handle,
                        make_skipped_item(dataset_name, episode, error),
                    )
                    processed_count += 1
                    dropped_count += 1
                    progress.update(1)
                    progress.set_postfix_str(
                        f"done={processed_count}/{total_selected_count}, "
                        f"skipped={num_skipped}, dropped={dropped_count}, "
                        f"ep={episode.episode_id}, rollout=failed"
                    )
                    continue

                append_jsonl_item(progress_handle, item)
                processed_count += 1
                progress.update(1)
                progress.set_postfix_str(
                    f"done={processed_count}/{total_selected_count}, "
                    f"skipped={num_skipped}, dropped={dropped_count}, "
                    f"ep={item['episode_id']}, actions={len(item['actions'])}, "
                    f"{time.time() - episode_start_time:.2f}s"
                )
    finally:
        if env is not None:
            with silence_external_output():
                env.close()


def preprocess_worker(
    result_queue,
    dataset_name,
    goal_radius,
    episode_ids,
    partial_output_path,
    partial_skipped_output_path,
    worker_index,
    local_gpu_id,
    display_gpu_id,
    process_index_on_gpu,
):
    env = None

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_gpu_id)

        env_config, dataset = load_dataset(
            dataset_name=dataset_name,
        )
        configure_action_only_env(env_config, local_gpu_id=local_gpu_id)

        selected_episode_ids = {int(episode_id) for episode_id in episode_ids}
        dataset.episodes = [
            episode
            for episode in dataset.episodes
            if int(episode.episode_id) in selected_episode_ids
        ]
        found_episode_ids = {int(episode.episode_id) for episode in dataset.episodes}
        missing_episode_ids = sorted(selected_episode_ids - found_episode_ids)
        if missing_episode_ids:
            raise ValueError(
                f"[{dataset_name}] worker {worker_index} missing episode ids: "
                f"{missing_episode_ids}"
            )

        dataset.episodes = sort_episodes_for_scene_locality(dataset.episodes)

        with silence_external_output():
            env = habitat.Env(config=env_config.habitat, dataset=dataset)

        os.makedirs(os.path.dirname(partial_output_path), exist_ok=True)
        with open(partial_output_path, "a", encoding="utf-8") as handle, open(
            partial_skipped_output_path, "a", encoding="utf-8"
        ) as skipped_handle:
            for episode in dataset.episodes:
                episode_start_time = time.time()
                try:
                    item = rollout_shortest_path_episode(
                        env=env,
                        episode=episode,
                        goal_radius=goal_radius,
                    )
                except ShortestPathRolloutError as error:
                    append_jsonl_item(
                        skipped_handle,
                        make_skipped_item(dataset_name, episode, error),
                    )
                    result_queue.put(
                        {
                            "status": "skipped",
                            "worker_index": worker_index,
                            "display_gpu_id": display_gpu_id,
                            "process_index_on_gpu": process_index_on_gpu,
                            "episode_id": int(episode.episode_id),
                            "time_per_episode": time.time() - episode_start_time,
                            "error": str(error),
                        }
                    )
                    continue

                append_jsonl_item(handle, item)
                result_queue.put(
                    {
                        "status": "ok",
                        "worker_index": worker_index,
                        "display_gpu_id": display_gpu_id,
                        "process_index_on_gpu": process_index_on_gpu,
                        "episode_id": int(episode.episode_id),
                        "time_per_episode": time.time() - episode_start_time,
                    }
                )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "worker_index": worker_index,
                "display_gpu_id": display_gpu_id,
                "process_index_on_gpu": process_index_on_gpu,
                "error": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            with silence_external_output():
                env.close()


def terminate_processes(process_jobs):
    for job in process_jobs:
        process = job["process"]
        if process.is_alive():
            process.terminate()
    for job in process_jobs:
        job["process"].join()


def process_dataset(
    dataset_name,
    output_root,
    goal_radius,
    requested_gpu_ids,
    visible_gpu_ids,
    num_processes_per_gpu,
    skip_existing_episodes,
    episode_ids,
    max_episodes,
    temp_root,
):
    output_path = default_output_path(output_root, dataset_name)
    skipped_path = skipped_output_path(output_path)
    progress_dir = progress_dir_path(output_path, temp_root=temp_root)

    if not skip_existing_episodes:
        remove_if_exists(output_path)
        remove_if_exists(skipped_path)
        remove_if_exists(progress_dir)

    existing_annotation_index = {}
    if skip_existing_episodes:
        existing_annotation_index.update(load_annotation_index(output_path))
        existing_annotation_index.update(load_annotation_index(skipped_path))
        existing_annotation_index.update(
            load_annotation_index_from_paths(
                list_rank_output_paths(progress_dir),
                tolerate_partial=True,
            )
        )
        existing_annotation_index.update(
            load_annotation_index_from_paths(
                list_rank_skipped_output_paths(progress_dir),
                tolerate_partial=True,
            )
        )

    env_config, dataset = load_dataset(
        dataset_name=dataset_name,
    )
    all_selected_episodes = filter_episodes(
        dataset.episodes,
        episode_ids=episode_ids,
        max_episodes=max_episodes,
    )

    if not all_selected_episodes:
        if skip_existing_episodes and (
            os.path.exists(output_path) or os.path.isdir(progress_dir)
        ):
            finalize_preprocess_outputs(
                output_path=output_path,
                progress_dir=progress_dir,
                merge_existing_output=True,
            )
        else:
            write_jsonl(output_path, [])
            write_jsonl(skipped_path, [])
        return

    selected_episode_ids = {int(episode.episode_id) for episode in all_selected_episodes}
    completed_episode_ids = selected_episode_ids & set(existing_annotation_index.keys())
    selected_episodes = [
        episode
        for episode in all_selected_episodes
        if int(episode.episode_id) not in existing_annotation_index
    ]
    total_selected_count = len(all_selected_episodes)
    num_skipped = len(completed_episode_ids)

    if not selected_episodes:
        finalize_preprocess_outputs(
            output_path=output_path,
            progress_dir=progress_dir,
            merge_existing_output=True,
        )
        tqdm.write(
            f"[preprocess:{dataset_name}] all {total_selected_count} selected episodes "
            f"already exist, skip everything"
        )
        return

    scene_count = len(group_episodes_by_scene(selected_episodes))
    configured_worker_assignments = build_worker_assignments(
        requested_gpu_ids=requested_gpu_ids,
        visible_gpu_ids=visible_gpu_ids,
        num_processes_per_gpu=num_processes_per_gpu,
    )
    max_workers = min(
        len(configured_worker_assignments),
        len(selected_episodes),
    )
    active_worker_assignments = configured_worker_assignments[:max_workers]
    os.makedirs(progress_dir, exist_ok=True)

    if max_workers <= 1:
        generate_annotations_sequential(
            env_config=env_config,
            dataset=dataset,
            selected_episodes=selected_episodes,
            dataset_name=dataset_name,
            goal_radius=goal_radius,
            progress_output_path=rank_output_path(progress_dir, 0),
            skipped_progress_output_path=rank_skipped_output_path(progress_dir, 0),
            total_selected_count=total_selected_count,
            num_skipped=num_skipped,
        )
        finalize_preprocess_outputs(
            output_path=output_path,
            progress_dir=progress_dir,
            merge_existing_output=skip_existing_episodes,
        )
        return

    worker_plans = build_worker_plans(
        selected_episodes=selected_episodes,
        worker_assignments=active_worker_assignments,
    )
    assignment_summary = [
        (
            f"gpu{plan['assignment']['display_gpu_id']}:"
            f"p{plan['assignment']['process_index_on_gpu']} "
            f"scenes={len(plan['scene_ids'])} eps={plan['episode_count']}"
        )
        for plan in worker_plans
    ]
    tqdm.write(
        f"[preprocess:{dataset_name}] pending={len(selected_episodes)}, "
        f"skipped={num_skipped}, total={total_selected_count}, "
        f"scenes={scene_count}, workers={len(worker_plans)}, "
        f"assignments={assignment_summary}"
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process_jobs = []

    total_bar = tqdm(
        total=total_selected_count,
        desc=f"preprocess:{dataset_name} total",
        position=0,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    if num_skipped:
        total_bar.update(num_skipped)
        total_bar.set_postfix_str(
            f"done={num_skipped}/{total_selected_count}, skipped={num_skipped}"
        )

    worker_bars = {}

    for position, plan in enumerate(worker_plans, start=1):
        assignment = plan["assignment"]
        partial_output_path = rank_output_path(progress_dir, assignment["worker_index"])
        partial_skipped_output_path = rank_skipped_output_path(
            progress_dir, assignment["worker_index"]
        )
        process = ctx.Process(
            target=preprocess_worker,
            args=(
                result_queue,
                dataset_name,
                goal_radius,
                plan["episode_ids"],
                partial_output_path,
                partial_skipped_output_path,
                assignment["worker_index"],
                assignment["local_gpu_id"],
                assignment["display_gpu_id"],
                assignment["process_index_on_gpu"],
            ),
        )
        process.start()
        process_jobs.append(
            {
                "process": process,
                "assignment": assignment,
                "partial_output_path": partial_output_path,
                "partial_skipped_output_path": partial_skipped_output_path,
            }
        )
        worker_bars[assignment["worker_index"]] = tqdm(
            total=plan["episode_count"],
            desc=(
                f"{dataset_name} gpu{assignment['display_gpu_id']}"
                f" p{assignment['process_index_on_gpu']}"
            ),
            position=position,
            dynamic_ncols=True,
            file=sys.stdout,
            leave=True,
        )
    
    processed_episodes = num_skipped
    dropped_episodes = 0
    success = False
    try:
        while processed_episodes < total_selected_count:
            try:
                result = result_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                crashed_jobs = [
                    job
                    for job in process_jobs
                    if not job["process"].is_alive()
                    and job["process"].exitcode not in (None, 0)
                ]
                if crashed_jobs:
                    raise RuntimeError(
                        f"[preprocess:{dataset_name}] worker crashed: "
                        f"{[(job['assignment']['worker_index'], job['process'].exitcode) for job in crashed_jobs]}"
                    )
                continue

            if result["status"] == "error":
                raise RuntimeError(
                    f"[preprocess:{dataset_name}] worker "
                    f"gpu{result['display_gpu_id']} "
                    f"p{result['process_index_on_gpu']} failed:\n"
                    f"{result['error']}"
                )

            if result["status"] == "skipped":
                dropped_episodes += 1

            processed_episodes += 1
            total_bar.update(1)
            total_bar.set_postfix_str(
                f"done={processed_episodes}/{total_selected_count}, "
                f"skipped={num_skipped}, dropped={dropped_episodes}"
            )

            worker_bar = worker_bars[result["worker_index"]]
            worker_bar.update(1)
            if result["status"] == "skipped":
                worker_bar.set_postfix_str(
                    f"ep={result['episode_id']} dropped, "
                    f"{result['time_per_episode']:.2f}s"
                )
            else:
                worker_bar.set_postfix_str(
                    f"ep={result['episode_id']}, {result['time_per_episode']:.2f}s"
                )

        for job in process_jobs:
            job["process"].join()
            if job["process"].exitcode != 0:
                raise RuntimeError(
                    f"[preprocess:{dataset_name}] worker "
                    f"{job['assignment']['worker_index']} exited with "
                    f"code {job['process'].exitcode}"
                )

        success = True
    finally:
        if not success:
            terminate_processes(process_jobs)
        else:
            for job in process_jobs:
                if job["process"].is_alive():
                    job["process"].join()
        total_bar.close()
        for worker_bar in worker_bars.values():
            worker_bar.close()

    finalize_preprocess_outputs(
        output_path=output_path,
        progress_dir=progress_dir,
        merge_existing_output=skip_existing_episodes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r"],
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/workspace/code_dir/a_property/dataset/PanoVLN",
        help="Root directory used to write generated sub_dataset jsonl files.",
    )
    parser.add_argument(
        "--goal_radius",
        type=float,
        default=DEFAULT_GOAL_RADIUS,
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Physical or visible GPU ids, e.g. '6,7' or '0 1'",
    )
    parser.add_argument(
        "--num_processes_per_gpu",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--skip_existing_episodes",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "--episode_ids",
        nargs="*",
        default=None,
        help="Optional episode ids to generate, e.g. --episode_ids 7142 9001",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--temp_root",
        type=str,
        default=None,
        help="Optional temp directory for preprocess in-progress files.",
    )

    args = parser.parse_args()
    selected_episode_ids = parse_episode_ids(args.episode_ids)
    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)

    visible_gpu_ids = None
    if requested_gpu_ids is not None:
        visible_gpu_ids = remap_gpu_ids_to_visible_devices(requested_gpu_ids)
        validate_gpu_ids(visible_gpu_ids)

    if args.num_processes_per_gpu is not None and requested_gpu_ids is None:
        raise ValueError("--num_processes_per_gpu requires --gpu_ids")

    for dataset_name in args.dataset_name:
        if dataset_name not in CONFIG:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        output_path = default_output_path(args.output_root, dataset_name)
        print(
            f"processing {dataset_name} with yaml dataset config -> {output_path}"
        )
        process_dataset(
            dataset_name=dataset_name,
            output_root=args.output_root,
            goal_radius=args.goal_radius,
            requested_gpu_ids=requested_gpu_ids,
            visible_gpu_ids=visible_gpu_ids,
            num_processes_per_gpu=args.num_processes_per_gpu,
            skip_existing_episodes=args.skip_existing_episodes,
            episode_ids=selected_episode_ids,
            max_episodes=args.max_episodes,
            temp_root=args.temp_root,
        )


if __name__ == "__main__":
    main()

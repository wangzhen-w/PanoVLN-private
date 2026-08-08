import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.habitat_shortest_path import (
    CONFIG,
    ERP_IMAGE_SIZE,
    STOP_ACTION,
    build_locality_balanced_episode_splits,
    build_worker_assignments,
    count_saved_frames,
    frame_image_filename,
    habitat,
    load_dataset,
    parse_episode_ids,
    parse_gpu_ids,
    remap_gpu_ids_to_visible_devices,
    reset_episode_output_dir,
    resolve_dataset_paths,
    save_rgb_frame,
    normalize_image_format,
    sort_episodes_for_scene_locality,
    silence_external_output,
    validate_gpu_ids,
)

SCAN_PROGRESS_REFRESH_INTERVAL_EPISODES = 256
QUEUE_POLL_TIMEOUT_SECONDS = 5


def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def expected_frame_count(annotation):
    return len(annotation["actions"])


def annotation_image_id(annotation):
    image_id = annotation.get("trajectory_id")
    if image_id is None:
        image_id = annotation["episode_id"]
    return str(image_id)


def is_episode_complete(image_path, annotation, image_format):
    episode_image_path = os.path.join(image_path, annotation_image_id(annotation))
    return (
        count_saved_frames(episode_image_path, image_format=image_format)
        == expected_frame_count(annotation)
    )


def replay_annotation_episode(
    env,
    episode,
    annotation,
    episode_image_path=None,
    image_size=ERP_IMAGE_SIZE,
    image_format="jpeg",
    jpeg_quality=75,
    jpeg_subsampling=2,
    png_compress_level=6,
):
    env.current_episode = episode
    observation = env.reset()
    actions = [int(action) for action in annotation["actions"]]

    if not actions:
        raise RuntimeError(f"Episode {episode.episode_id} has empty action list")
    if actions[-1] != STOP_ACTION:
        raise RuntimeError(
            f"Episode {episode.episode_id} action list must end with stop"
        )

    step_id = 0
    if episode_image_path is not None:
        reset_episode_output_dir(episode_image_path)
        save_rgb_frame(
            observation["rgb"],
            os.path.join(
                episode_image_path,
                frame_image_filename(0, image_format=image_format),
            ),
            image_size=image_size,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            png_compress_level=png_compress_level,
        )

    for action_index, action in enumerate(actions):
        if action == STOP_ACTION and action_index != len(actions) - 1:
            raise RuntimeError(
                f"Episode {episode.episode_id} contains stop before the final action"
            )

        observation = env.step(action)
        if action == STOP_ACTION:
            continue

        step_id += 1
        if episode_image_path is not None:
            save_rgb_frame(
                observation["rgb"],
                os.path.join(
                    episode_image_path,
                    frame_image_filename(step_id, image_format=image_format),
                ),
                image_size=image_size,
                jpeg_quality=jpeg_quality,
                jpeg_subsampling=jpeg_subsampling,
                png_compress_level=png_compress_level,
            )

    return {
        "episode_id": int(episode.episode_id),
        "frame_count": step_id + 1,
    }


def extract_data(
        result_queue,
        dataset_name,
        annotations,
        episode_ids,
        worker_index,
        save_image=False,
        image_path=None,
        local_gpu_id=0,
        display_gpu_id=0,
        process_index_on_gpu=0,
        image_format="jpeg",
        jpeg_quality=75,
        jpeg_subsampling=2,
        png_compress_level=6,
    ):
    env = None

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_gpu_id)

        env_config, dataset = load_dataset(
            dataset_name=dataset_name,
        )
        with habitat.config.read_write(env_config):
            env_config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_gpu_id

        selected_episode_ids = {int(episode_id) for episode_id in episode_ids}
        dataset.episodes = [
            episode
            for episode in dataset.episodes
            if int(episode.episode_id) in selected_episode_ids
        ]
        dataset.episodes = sort_episodes_for_scene_locality(dataset.episodes)

        found_episode_ids = {int(episode.episode_id) for episode in dataset.episodes}
        missing_episode_ids = sorted(selected_episode_ids - found_episode_ids)
        if missing_episode_ids:
            raise ValueError(
                f"[{dataset_name}] worker {worker_index} missing episode ids: "
                f"{missing_episode_ids}"
            )

        annotation_by_episode_id = {
            int(annotation["episode_id"]): annotation for annotation in annotations
        }
        missing_annotations = sorted(found_episode_ids - set(annotation_by_episode_id.keys()))
        if missing_annotations:
            raise ValueError(
                f"[{dataset_name}] worker {worker_index} missing annotations for "
                f"episode ids: {missing_annotations}"
            )
        ordered_annotations = [
            annotation_by_episode_id[int(episode.episode_id)]
            for episode in dataset.episodes
        ]

        with silence_external_output():
            env = habitat.Env(config=env_config.habitat, dataset=dataset)
            if save_image:
                os.makedirs(image_path, exist_ok=True)

            if len(ordered_annotations) != len(env.episodes):
                raise ValueError(
                    f"[{dataset_name}] worker {worker_index} annotation/env mismatch: "
                    f"{len(ordered_annotations)} vs {len(env.episodes)}"
                )

            for idx, episode in enumerate(env.episodes):
                output_dict = {
                    "status": "ok",
                    "worker_index": worker_index,
                    "display_gpu_id": display_gpu_id,
                    "process_index_on_gpu": process_index_on_gpu,
                    "time_per_episode": 0,
                }
                episode_start_time = time.time()
                annotation = ordered_annotations[idx]
                if int(annotation["episode_id"]) != int(episode.episode_id):
                    raise ValueError(
                        f"[{dataset_name}] worker {worker_index} episode order mismatch: "
                        f"annotation={annotation['episode_id']} env={episode.episode_id}"
                    )
                current_image_path = None
                if save_image:
                    current_image_path = os.path.join(
                        image_path, annotation_image_id(annotation)
                    )
                replay_result = replay_annotation_episode(
                    env=env,
                    episode=episode,
                    annotation=annotation,
                    episode_image_path=current_image_path,
                    image_size=CONFIG[dataset_name].get(
                        "image_size", ERP_IMAGE_SIZE
                    ),
                    image_format=image_format,
                    jpeg_quality=jpeg_quality,
                    jpeg_subsampling=jpeg_subsampling,
                    png_compress_level=png_compress_level,
                )
                if replay_result["frame_count"] != expected_frame_count(annotation):
                    raise RuntimeError(
                        f"Episode {episode.episode_id} saved "
                        f"{replay_result['frame_count']} frames, expected "
                        f"{expected_frame_count(annotation)}"
                    )
                output_dict["time_per_episode"] = time.time() - episode_start_time
                output_dict["episode_id"] = int(episode.episode_id)
                result_queue.put(output_dict)

            env.close()
            env = None
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


def process_single_dataset(
    dataset_name,
    output_root,
    save_image,
    num_thread,
    requested_gpu_ids,
    visible_gpu_ids,
    num_processes_per_gpu,
    skip_existing_episodes,
    max_episodes,
    episode_ids,
    image_format,
    jpeg_quality,
    jpeg_subsampling,
    png_compress_level,
):
    ANNOT_PATH, IMAGE_PATH = resolve_dataset_paths(
        dataset_name,
        output_root,
    )
    annotations = []
    with open(ANNOT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            annotations.append(item)
    annotations = sorted(annotations, key=lambda x: x["episode_id"])
    if episode_ids is not None:
        selected_episode_id_set = {int(episode_id) for episode_id in episode_ids}
        annotations = [
            annotation
            for annotation in annotations
            if annotation["episode_id"] in selected_episode_id_set
        ]
        missing_episode_ids = sorted(
            selected_episode_id_set
            - {annotation["episode_id"] for annotation in annotations}
        )
        if missing_episode_ids:
            raise ValueError(
                f"Missing sub_dataset annotations for episode ids: {missing_episode_ids}"
            )

    if max_episodes is not None:
        annotations = annotations[:max_episodes]

    _, dataset = load_dataset(
        dataset_name=dataset_name,
    )
    dataset.episodes = sorted(
        dataset.episodes, key=lambda episode: int(episode.episode_id)
    )

    selected_episode_ids = {
        annotation["episode_id"] for annotation in annotations
    }
    dataset.episodes = [
        episode
        for episode in dataset.episodes
        if int(episode.episode_id) in selected_episode_ids
    ]
    dataset_episode_ids = {int(episode.episode_id) for episode in dataset.episodes}

    annotations = [
        annotation
        for annotation in annotations
        if annotation["episode_id"] in dataset_episode_ids
    ]
    annotation_by_episode_id = {
        annotation["episode_id"]: annotation for annotation in annotations
    }
    if len(dataset.episodes) != len(annotations):
        annotation_episode_ids = {annotation["episode_id"] for annotation in annotations}
        missing_in_dataset = sorted(annotation_episode_ids - dataset_episode_ids)
        missing_in_annotations = sorted(dataset_episode_ids - annotation_episode_ids)
        raise ValueError(
            f"Dataset/annotation mismatch for {dataset_name}: "
            f"missing_in_dataset={missing_in_dataset}, "
            f"missing_in_annotations={missing_in_annotations}"
        )

    num_episodes = len(dataset.episodes)
    num_skipped = 0
    if save_image and skip_existing_episodes:
        pending_episodes = []
        pending_annotations = []
        scan_bar = tqdm(
            total=num_episodes,
            desc=f"{dataset_name} scan",
            position=0,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for checked_count, episode in enumerate(dataset.episodes, start=1):
            annotation = annotation_by_episode_id[int(episode.episode_id)]
            if is_episode_complete(IMAGE_PATH, annotation, image_format):
                num_skipped += 1
            else:
                pending_episodes.append(episode)
                pending_annotations.append(annotation)

            should_refresh = (
                checked_count % SCAN_PROGRESS_REFRESH_INTERVAL_EPISODES == 0
                or checked_count == num_episodes
            )
            if should_refresh:
                scan_bar.update(checked_count - scan_bar.n)
                scan_bar.set_postfix_str(
                    f"checked={checked_count}/{num_episodes}, skipped={num_skipped}"
                )
        scan_bar.close()
        dataset.episodes = pending_episodes
        annotations = pending_annotations
        annotation_by_episode_id = {
            annotation["episode_id"]: annotation for annotation in annotations
        }

    if not dataset.episodes:
        tqdm.write(
            f"[{dataset_name}] all {num_episodes} episodes already exist, skip everything"
        )
        return

    worker_assignments = build_worker_assignments(
        requested_gpu_ids=requested_gpu_ids,
        visible_gpu_ids=visible_gpu_ids,
        num_processes_per_gpu=num_processes_per_gpu,
        max_workers=num_thread,
    )
    worker_assignments = worker_assignments[: min(len(worker_assignments), len(dataset.episodes))]
    episode_splits = build_locality_balanced_episode_splits(
        dataset.episodes,
        len(worker_assignments),
    )

    worker_jobs = []
    for worker_assignment, worker_episodes in zip(worker_assignments, episode_splits):
        if not worker_episodes:
            continue

        worker_episode_ids = [int(episode.episode_id) for episode in worker_episodes]
        worker_annotations = [
            annotation_by_episode_id[int(episode.episode_id)]
            for episode in worker_episodes
        ]
        if worker_annotations:
            worker_jobs.append(
                {
                    "assignment": worker_assignment,
                    "episode_ids": worker_episode_ids,
                    "annotations": worker_annotations,
                }
            )

    total_bar = tqdm(
        total=num_episodes,
        desc=f"{dataset_name} total",
        position=0,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    if num_skipped:
        total_bar.update(num_skipped)
        total_bar.set_postfix_str(
            f"done={total_bar.n}/{num_episodes}, skipped={num_skipped}"
        )

    worker_labels = [
        f"gpu{job['assignment']['display_gpu_id']}:p{job['assignment']['process_index_on_gpu']}"
        for job in worker_jobs
    ]
    tqdm.write(
        f"[{dataset_name}] total={num_episodes}, skipped={num_skipped}, "
        f"workers={len(worker_jobs)}, assignments={worker_labels}"
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for job in worker_jobs:
        assignment = job["assignment"]
        worker_episode_ids = job["episode_ids"]
        worker_annotations = job["annotations"]
        worker_args = (
            result_queue,
            dataset_name,
            worker_annotations,
            worker_episode_ids,
            assignment["worker_index"],
            save_image,
            IMAGE_PATH,
            assignment["local_gpu_id"],
            assignment["display_gpu_id"],
            assignment["process_index_on_gpu"],
            image_format,
            jpeg_quality,
            jpeg_subsampling,
            png_compress_level,
        )
        p = ctx.Process(target=extract_data, args=worker_args)
        p.start()
        processes.append(p)

    worker_bars = {}

    for position, job in enumerate(worker_jobs, start=1):
        assignment = job["assignment"]
        worker_bars[assignment["worker_index"]] = tqdm(
            total=len(job["annotations"]),
            desc=(
                f"{dataset_name} gpu{assignment['display_gpu_id']}"
                f" p{assignment['process_index_on_gpu']}"
            ),
            position=position,
            dynamic_ncols=True,
            file=sys.stdout,
            leave=True,
        )

    total_bar.set_postfix_str(
        f"done={total_bar.n}/{num_episodes}, skipped={num_skipped}"
    )

    try:
        for _ in range(num_episodes - num_skipped):
            try:
                result = result_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                crashed_processes = [
                    process
                    for process in processes
                    if not process.is_alive() and process.exitcode not in (None, 0)
                ]
                if crashed_processes:
                    raise RuntimeError(
                        f"[{dataset_name}] worker crashed: "
                        f"{[(process.pid, process.exitcode) for process in crashed_processes]}"
                    )
                continue

            if result["status"] == "error":
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join()
                raise RuntimeError(
                    f"[{dataset_name}] worker gpu{result['display_gpu_id']} "
                    f"p{result['process_index_on_gpu']} failed:\n{result['error']}"
                )

            total_bar.update(1)
            total_bar.set_postfix_str(
                f"done={total_bar.n}/{num_episodes}, skipped={num_skipped}"
            )

            worker_bar = worker_bars[result["worker_index"]]
            worker_bar.update(1)
            worker_bar.set_postfix_str(
                f"ep={result['episode_id']}, {result['time_per_episode']:.2f}s"
            )
    finally:
        for process in processes:
            process.join()
        total_bar.close()
        for worker_bar in worker_bars.values():
            worker_bar.close()

def main(
    dataset2process,
    output_root,
    save_image,
    num_thread,
    requested_gpu_ids,
    visible_gpu_ids,
    num_processes_per_gpu,
    skip_existing_episodes,
    max_episodes,
    episode_ids,
    image_format,
    jpeg_quality,
    jpeg_subsampling,
    png_compress_level,
):
    for dataset_name in dataset2process:
        process_single_dataset(
            dataset_name=dataset_name,
            output_root=output_root,
            save_image=save_image,
            num_thread=num_thread,
            requested_gpu_ids=requested_gpu_ids,
            visible_gpu_ids=visible_gpu_ids,
            num_processes_per_gpu=num_processes_per_gpu,
            skip_existing_episodes=skip_existing_episodes,
            max_episodes=max_episodes,
            episode_ids=episode_ids,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            png_compress_level=png_compress_level,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r",],
    )
    parser.add_argument(
        "--save_image",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
    )
    parser.add_argument("--num_thread", type=int, default=16)
    parser.add_argument(
        "--output_root",
        type=str,
        default="/workspace/data2/dataset/PanoVLN",
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
        default=True,
        type=str2bool,
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--episode_ids",
        nargs="*",
        default=None,
        help="Optional episode ids to render, e.g. --episode_ids 7142 9001",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="jpeg",
        choices=("jpg", "jpeg", "png"),
        help="Output image format. 'jpg' is an alias for 'jpeg'.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=75)
    parser.add_argument(
        "--jpeg_subsampling",
        type=int,
        default=2,
        choices=(0, 1, 2),
    )
    parser.add_argument(
        "--png_compress_level",
        type=int,
        default=6,
        choices=tuple(range(10)),
        help="PNG DEFLATE level (0 disables compression; all levels are lossless).",
    )

    args = parser.parse_args()
    image_format = normalize_image_format(args.image_format)
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError(
            f"--jpeg_quality must be in [1, 100], got {args.jpeg_quality}"
        )
    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
    selected_episode_ids = parse_episode_ids(args.episode_ids)
    visible_gpu_ids = None
    if requested_gpu_ids is not None:
        visible_gpu_ids = remap_gpu_ids_to_visible_devices(requested_gpu_ids)
        validate_gpu_ids(visible_gpu_ids)

    if args.num_processes_per_gpu is not None and requested_gpu_ids is None:
        raise ValueError("--num_processes_per_gpu requires --gpu_ids")

    main(
        dataset2process=args.dataset_name,
        output_root=args.output_root,
        save_image=args.save_image,
        num_thread=args.num_thread,
        requested_gpu_ids=requested_gpu_ids,
        visible_gpu_ids=visible_gpu_ids,
        num_processes_per_gpu=args.num_processes_per_gpu,
        skip_existing_episodes=args.skip_existing_episodes,
        max_episodes=args.max_episodes,
        episode_ids=selected_episode_ids,
        image_format=image_format,
        jpeg_quality=args.jpeg_quality,
        jpeg_subsampling=args.jpeg_subsampling,
        png_compress_level=args.png_compress_level,
    )

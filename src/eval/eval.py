import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence
import torch
import numpy as np
# Import torch before habitat to avoid CUDA runtime conflicts during module loading.
from habitat import Env
from habitat.core.agent import Agent
from tqdm import tqdm
import os
import re
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
import random
from transformers import AutoProcessor
import argparse, habitat
from habitat_extensions import measures, task
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image
from peft import PeftModel
from src.train.data.data import (
    DEFAULT_ERP_BOTTOM_CROP_DEGREES,
    DEFAULT_ERP_TOP_CROP_DEGREES,
    DEFAULT_VLN_MAX_MEMORY_IMAGES as DEFAULT_MAX_MEMORY_IMAGES,
    DEFAULT_VLN_MEMORY_POOL_WINDOW_FRAMES as DEFAULT_MEMORY_POOL_WINDOW_FRAMES,
    VLN_SYSTEM_PROMPT,
    build_erp_image_geometry_batch,
    build_vln_image_selection,
    build_vln_user_content,
    preprocess_panovggt_current_image,
    preprocess_vln_current_image,
    preprocess_vln_memory_image,
    resolve_current_image_index,
)
from src.qwen_vl import Qwen3_5ForConditionalGenerationForPanoVLN
from src.train.utils import build_prompt_and_target

SYSTEM_PROMPT = VLN_SYSTEM_PROMPT
TARGET_KEYS = ("success", "spl", "oracle_success", "distance_to_goal", "path_length", "ndtw")
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-\d+$")
DEFAULT_EVAL_GENERATION_KWARGS = {
    "max_new_tokens": 24,
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
}

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.getLogger("imageio.plugins.ffmpeg").setLevel(logging.ERROR)

ATOMIC_ACTION_NAMES = ("stop", "forward", "left", "right")
ATOMIC_ACTION_TO_ID = {action_name: action_id for action_id, action_name in enumerate(ATOMIC_ACTION_NAMES)}
STOP_ACTION_ID = ATOMIC_ACTION_TO_ID["stop"]
ACTION_SEQUENCE_LENGTH = 4
ATOMIC_ACTION_VARIANTS = {
    "stop": ("stop",),
    "forward": ("forward", "move_forward", "move forward", "move-forward"),
    "left": ("left", "turn_left", "turn left", "turn-left"),
    "right": ("right", "turn_right", "turn right", "turn-right"),
}
ATOMIC_ACTION_PATTERNS = [
    (
        action_id,
        re.compile(
            r"(?:"
            + "|".join(
                r"(?<![0-9a-z_])" + re.escape(variant.lower()) + r"(?![0-9a-z_])"
                for variant in ATOMIC_ACTION_VARIANTS[action_name]
            )
            + r")"
        ),
    )
    for action_id, action_name in enumerate(ATOMIC_ACTION_NAMES)
]


def seed_all(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_eval_model_path(model_path: str) -> str:
    resolved_model_path = str(Path(model_path).expanduser())
    if CHECKPOINT_DIR_PATTERN.fullmatch(Path(resolved_model_path).name):
        raise ValueError(
            "Eval model_path must point to a fully saved model directory, "
            f"not an intermediate Trainer checkpoint: {resolved_model_path}"
        )
    return resolved_model_path


def build_eval_messages(instruction: str, images: List[Image.Image]):
    user_content_template = build_vln_user_content(
        instruction=instruction,
        num_images=len(images),
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }
    ]

    messages.append(
        {
            "role": "user",
            "content": user_content_template,
        }
    )
    return messages


def build_eval_generation_prompt(processor, messages: List[Dict]) -> str:
    prompt_and_target = build_prompt_and_target(
        messages,
        prompt_format="chat_template",
        processor=processor,
        require_target=False,
    )
    return prompt_and_target["prompt"]


def select_vln_eval_image_indices(
    history_length: int,
    max_memory_images: int,
    memory_pool_window_frames: int,
    pbo_enabled: bool = False,
) -> List[int]:
    last_frame_index = history_length - 1
    if last_frame_index < 0:
        return []
    return build_vln_image_selection(
        current_step=last_frame_index,
        last_frame_index=last_frame_index,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
        required_frame_indices=(
            [last_frame_index - ACTION_SEQUENCE_LENGTH]
            if pbo_enabled and last_frame_index >= ACTION_SEQUENCE_LENGTH
            else None
        ),
    )


def preprocess_vln_eval_images(
    rgb_history: Sequence[Image.Image],
    selected_indices: Sequence[int],
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
) -> List[Image.Image]:
    selected_images = []
    for image_position, frame_index in enumerate(selected_indices):
        raw_image = rgb_history[frame_index]
        is_current_observation = image_position == len(selected_indices) - 1
        if is_current_observation:
            selected_images.append(
                preprocess_vln_current_image(
                    raw_image,
                    top_crop_degrees=top_crop_degrees,
                    bottom_crop_degrees=bottom_crop_degrees,
                )
            )
        else:
            selected_images.append(
                preprocess_vln_memory_image(
                    raw_image,
                    top_crop_degrees=top_crop_degrees,
                    bottom_crop_degrees=bottom_crop_degrees,
                )
            )
    return selected_images


def parse_action_sequence(output: str, max_actions: int = ACTION_SEQUENCE_LENGTH):
    if "</think>" in output.lower():
        output = re.split(r"</think>", output, flags=re.IGNORECASE)[-1]

    action_text = " ".join(output.split()).lower().strip(" \t\r\n`'\".,;:!?()[]{}")
    if not action_text:
        return []

    matches = []
    for action_id, pattern in ATOMIC_ACTION_PATTERNS:
        for match in pattern.finditer(action_text):
            matches.append((match.start(), match.end(), action_id))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    actions = []
    last_end = -1
    for start, end, action_id in matches:
        if start < last_end:
            continue
        actions.append(action_id)
        last_end = end
        if len(actions) >= max_actions:
            break

    return actions


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _scene_id_from_path(scene_path):
    scene_path = Path(str(scene_path))
    return scene_path.parent.name or scene_path.stem


def _make_episode_key(scene_id, episode_id):
    return _scene_id_from_path(scene_id), str(episode_id)


def _coerce_json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _result_row(scene_id, episode_id, info):
    row = {
        "id": str(episode_id),
        "scene_id": _scene_id_from_path(scene_id),
    }
    for key in TARGET_KEYS:
        if key in info:
            row[key] = _coerce_json_value(info[key])
    return row


def _iter_result_paths(result_path):
    result_path = Path(result_path)
    paths = []
    merged_path = result_path / "result.jsonl"
    if merged_path.exists():
        paths.append(merged_path)
    paths.extend(sorted(result_path.glob("result_rank*.jsonl")))
    return paths


def _load_result_rows(result_path):
    row_map = {}
    for path in _iter_result_paths(result_path):
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode_id = row.get("id", row.get("episode_id"))
                scene_id = row.get("scene_id")
                if episode_id is None or scene_id is None:
                    continue
                row["id"] = str(episode_id)
                row["scene_id"] = _scene_id_from_path(scene_id)
                row_map[_make_episode_key(scene_id, episode_id)] = row
    return [row_map[key] for key in sorted(row_map.keys())]


def _load_done_pairs(result_path):
    return {
        _make_episode_key(row["scene_id"], row["id"])
        for row in _load_result_rows(result_path)
    }


def _filter_pending_episodes(episodes, done_pairs):
    return [
        episode
        for episode in episodes
        if _make_episode_key(episode.scene_id, episode.episode_id) not in done_pairs
    ]


def _append_result_row(result_path, split_id, row):
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    shard_path = result_path / f"result_rank{split_id}.jsonl"
    with shard_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_rows(rows):
    summary = {"num_episodes": len(rows)}
    for key in TARGET_KEYS:
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = float(sum(values) / len(values)) if values else 0.0
    return summary


def evaluate_agent(
    config,
    split_id,
    dataset,
    model_path,
    lora_path,
    result_path,
    forward_distance,
    turn_angle,
    max_memory_images,
    memory_pool_window_frames,
    save_topdown,
    attn_implementation,
    early_stop_max_steps,
) -> None:
    done_pairs = _load_done_pairs(result_path)
    pending_episodes = _filter_pending_episodes(list(dataset.episodes), done_pairs)
    dataset.episodes = pending_episodes

    if not pending_episodes:
        progress = tqdm(
            pending_episodes,
            desc=f"split {split_id}",
            position=split_id,
            dynamic_ncols=True,
        )
        progress.close()
        return

    env = Env(config.habitat, dataset)
    agent = PanoVLN_Agent(
        model_path,
        lora_path,
        result_path,
        forward_distance,
        turn_angle,
        max_memory_images,
        memory_pool_window_frames,
        save_topdown=save_topdown,
        attn_implementation=attn_implementation,
    )

    early_stop_max_steps = max(0, int(early_stop_max_steps))
    progress = tqdm(
        pending_episodes,
        desc=f"split {split_id}",
        position=split_id,
        dynamic_ncols=True,
    )
    try:
        executed_action_log = []
        for episode in progress:
            agent.reset()
            executed_action_log = []
            env.current_episode = episode
            obs = env.reset()
            iter_step = 0

            while not env.episode_over:
                info = env.get_metrics()
                action = agent.act(obs, info, env.current_episode.episode_id)

                if early_stop_max_steps > 0 and iter_step >= early_stop_max_steps:
                    action = {"action": 0}

                iter_step += 1
                executed_action_log.append(int(action["action"]))
                obs = env.step(action)

            info = env.get_metrics()
            agent.finalize_episode()
            result_row = _result_row(episode.scene_id, episode.episode_id, info)
            result_row["executed_action_history"] = executed_action_log
            result_row["model_generated_actions"] = list(agent.model_generated_actions)
            result_row["model_parsed_action_sequences"] = list(agent.model_parsed_action_sequences)
            _append_result_row(result_path, split_id, result_row)
            agent.reset()

            postfix = {}
            if "success" in result_row:
                postfix["success"] = result_row["success"]
            if "spl" in result_row:
                postfix["spl"] = f"{float(result_row['spl']):.3f}"
            if postfix:
                progress.set_postfix(**postfix)
    finally:
        env.close()

class PanoVLN_Agent(Agent):
    def __init__(
        self,
        model_path,
        lora_path,
        result_path,
        forward_distance,
        turn_angle,
        max_memory_images,
        memory_pool_window_frames,
        save_topdown=False,
        attn_implementation="sdpa",
    ):
        
        print("Initialize PanoVLN")
        
        self.result_path = result_path
        self.save_topdown = save_topdown
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.max_memory_images = max(0, int(max_memory_images))
        self.memory_pool_window_frames = max(1, int(memory_pool_window_frames))
        self.attn_implementation = attn_implementation
        os.makedirs(self.result_path, exist_ok=True)
        if self.save_topdown:
            os.makedirs(os.path.join(self.result_path, "top_down"), exist_ok=True)

        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = self.attn_implementation
        model_init_kwargs['torch_dtype'] = torch.bfloat16

        self.model = Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(
            model_path,
            **model_init_kwargs,
        )

        if lora_path is not None and lora_path!= '':
            print('Loading LoRA weights...')
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print('Merging LoRA weights...')
            self.model = self.model.merge_and_unload()
            print('Model is loaded...')

        self.erp_top_crop_degrees = float(
            getattr(self.model.config, "erp_top_crop_degrees", DEFAULT_ERP_TOP_CROP_DEGREES)
        )
        self.erp_bottom_crop_degrees = float(
            getattr(self.model.config, "erp_bottom_crop_degrees", DEFAULT_ERP_BOTTOM_CROP_DEGREES)
        )
        self.pbo_enabled = bool(getattr(self.model.config, "pbo_enabled", False))
        self.device = 'cuda'
        self.model.to(self.device)
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("Eval requires a tokenizer for generation token ids")
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)

        if self.eos_token_id is not None:
            setattr(self.model.config, "eos_token_id", self.eos_token_id)
            text_config = getattr(self.model.config, "text_config", None)
            if text_config is not None:
                setattr(text_config, "eos_token_id", self.eos_token_id)
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.eos_token_id = self.eos_token_id
        if self.pad_token_id is not None:
            setattr(self.model.config, "pad_token_id", self.pad_token_id)
            text_config = getattr(self.model.config, "text_config", None)
            if text_config is not None:
                setattr(text_config, "pad_token_id", self.pad_token_id)
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.pad_token_id = self.pad_token_id
        if self.bos_token_id is not None:
            setattr(self.model.config, "bos_token_id", self.bos_token_id)
            text_config = getattr(self.model.config, "text_config", None)
            if text_config is not None:
                setattr(text_config, "bos_token_id", self.bos_token_id)
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.bos_token_id = self.bos_token_id
        print(
            "Initialization Complete "
            f"(attn_implementation={self.attn_implementation}, "
            f"pbo_enabled={self.pbo_enabled})"
        )
        
        self.rgb_history = []
        self.current_images = []
        self.model_generated_actions = []
        self.model_parsed_action_sequences = []
        self.pending_action_queue = []
        self.topdown_frames = []
        self.conversations = []

        self.reset()


    def predict_inference(self, gen_kwargs=None):
        generation_kwargs = dict(DEFAULT_EVAL_GENERATION_KWARGS)
        if gen_kwargs is not None:
            generation_kwargs.update(gen_kwargs)

        texts = [build_eval_generation_prompt(self.processor, self.conversations)]

        prompt_inputs = self.processor(
            text=texts,
            images=self.current_images if self.current_images else None,
            return_tensors="pt",
            padding=True,
        )
        image_count = len(self.current_images)
        image_erp_geometry = build_erp_image_geometry_batch(
            image_count,
            top_crop_degrees=self.erp_top_crop_degrees,
            bottom_crop_degrees=self.erp_bottom_crop_degrees,
        )
        prompt_inputs["image_erp_geometry"] = image_erp_geometry
        prompt_inputs["image_num_images"] = torch.tensor([image_count], dtype=torch.long)
        prompt_inputs["image_current_index"] = torch.tensor(
            [resolve_current_image_index(image_count)],
            dtype=torch.long,
        )
        if bool(getattr(self.model.config, "panovggt_enabled", False)) and self.rgb_history:
            prompt_inputs["panovggt_pixel_values"] = preprocess_panovggt_current_image(
                self.rgb_history[-1]
            ).unsqueeze(0)

        prompt_inputs = prompt_inputs.to(self.device)
        with torch.inference_mode():
            cont = self.model.generate(
                **prompt_inputs,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                do_sample=True if generation_kwargs["temperature"] > 0 else False,
                temperature=generation_kwargs["temperature"],
                top_p=generation_kwargs["top_p"],
                num_beams=generation_kwargs["num_beams"],
                max_new_tokens=generation_kwargs["max_new_tokens"],
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(prompt_inputs.input_ids, cont)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answers = [s.lower().strip() for s in answers]
        return answers[0] if answers else ""

    def _select_image_indices(self):
        return select_vln_eval_image_indices(
            history_length=len(self.rgb_history),
            max_memory_images=self.max_memory_images,
            memory_pool_window_frames=self.memory_pool_window_frames,
            pbo_enabled=self.pbo_enabled,
        )

    def _prepare_selected_images(self, selected_indices):
        return preprocess_vln_eval_images(
            rgb_history=self.rgb_history,
            selected_indices=selected_indices,
            top_crop_degrees=self.erp_top_crop_degrees,
            bottom_crop_degrees=self.erp_bottom_crop_degrees,
        )

    def _predict_action_sequence_from_images(
        self,
        instruction,
        selected_images,
    ):
        self.current_images = selected_images
        self.conversations = build_eval_messages(
            instruction=instruction,
            images=selected_images,
        )

        navigation = self.predict_inference()
        self.model_generated_actions.append(navigation)
        action_ids = parse_action_sequence(navigation)
        self.model_parsed_action_sequences.append(list(action_ids))
        return navigation, action_ids

    def _build_pending_action_queue(self, action_ids):
        action_ids = list(action_ids[:ACTION_SEQUENCE_LENGTH])
        if not action_ids:
            return [STOP_ACTION_ID]
        if STOP_ACTION_ID in action_ids:
            return action_ids[:action_ids.index(STOP_ACTION_ID) + 1]
        return action_ids

    def finalize_episode(self):
        pass

    def reset(self):       
        if self.save_topdown and getattr(self, "episode_id", None) is not None and self.topdown_frames:
            images_to_video(
                images=list(self.topdown_frames),
                output_dir=os.path.join(self.result_path, "top_down"),
                video_name=f"ep_{self.episode_id}",
                fps=3,
                verbose=False,
                ffmpeg_log_level="error",
            )

        self.topdown_frames = []
        self.rgb_history = []
        self.current_images = []
        self.model_generated_actions = []
        self.model_parsed_action_sequences = []
        self.pending_action_queue = []
        self.conversations = []
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        if self.save_topdown and info.get("top_down_map") is not None:
            top_down_frame = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map"], observations["rgb"].shape[0]
            )
            render_frame = np.concatenate((observations["rgb"], top_down_frame), axis=1)
            self.topdown_frames.append(render_frame)

        rgb = observations["rgb"]
        self.rgb_history.append(Image.fromarray(rgb.astype('uint8')).convert('RGB'))

        if not self.pending_action_queue:
            selected_indices = self._select_image_indices()
            selected_images = self._prepare_selected_images(selected_indices)
            navigation, action_ids = self._predict_action_sequence_from_images(
                instruction=observations["instruction"]["text"],
                selected_images=selected_images,
            )

            if not action_ids:
                print(
                    f"[Warning] Failed to parse a valid action sequence from model output "
                    f"on episode {episode_id}: {navigation!r}. Defaulting to stop."
                )
            self.pending_action_queue = self._build_pending_action_queue(action_ids)

        action_id = self.pending_action_queue.pop(0)
        if action_id == STOP_ACTION_ID:
            self.pending_action_queue = []

        return {"action": action_id}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--split-id",type=int,required=True,help="chunks ID of evluation")
    parser.add_argument("--model-path",type=str,required=True,help="location of fully saved model weights")
    parser.add_argument("--lora-path",type=str,help="location of lora weights", default=None)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument(
        "--max-memory-images",
        type=int,
        default=DEFAULT_MAX_MEMORY_IMAGES,
        help="maximum number of history images to keep before the current observation",
    )
    parser.add_argument(
        "--memory-pool-window-frames",
        type=int,
        default=DEFAULT_MEMORY_POOL_WINDOW_FRAMES,
        help="recent frame window used by the history-memory selection policy",
    )
    parser.add_argument("--max-episodes",type=int,default=0,help="limit eval episodes per worker after splitting; 0 means all")
    parser.add_argument("--total-max-episodes", type=int, default=0,
                        help="limit total eval episodes before splitting across workers; 0 means all")
    parser.add_argument("--save-topdown",type=str2bool,default=False,help="save per-episode top-down videos")
    parser.add_argument("--attn-implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="attention backend used to load the model")
    parser.add_argument("--early-stop-max-steps", type=int, default=0,
                        help="optional hard cap on env steps per episode; 0 relies on habitat.environment.max_episode_steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed for python, numpy, and torch")
    args = parser.parse_args()

    seed_all(args.seed)
    args.model_path = validate_eval_model_path(args.model_path)

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        measurement_updates = {
            "collisions": CollisionsMeasurementConfig(),
        }
        if args.save_topdown:
            measurement_updates["top_down_map"] = TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=360,
                ),
            )
        config.habitat.task.measurements.update(measurement_updates)
            
    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    done_pairs = _load_done_pairs(args.result_path)
    dataset.episodes = _filter_pending_episodes(list(dataset.episodes), done_pairs)

    if args.total_max_episodes > 0:
        dataset.episodes = list(dataset.episodes)[:args.total_max_episodes]

    if dataset.episodes:
        dataset_split = dataset.get_splits(args.split_num, allow_uneven_splits=True)[args.split_id]
    else:
        dataset_split = dataset
    if args.max_episodes > 0:
        dataset_split.episodes = list(dataset_split.episodes)[:args.max_episodes]

    evaluate_agent(config, args.split_id, dataset_split, args.model_path, args.lora_path, args.result_path,
                args.forward_distance, args.turn_angle, args.max_memory_images,
                args.memory_pool_window_frames, args.save_topdown, args.attn_implementation,
                args.early_stop_max_steps)

if __name__ == "__main__":
    main()

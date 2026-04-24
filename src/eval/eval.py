import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
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
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
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
from src.data.prepare_training_data import (
    DEFAULT_MAX_MEMORY_IMAGES,
    DEFAULT_MEMORY_POOL_WINDOW_FRAMES,
    VLN_SYSTEM_PROMPT,
    build_vln_image_selection,
    build_vln_user_content,
)
from src.train.data.data import (
    preprocess_vln_current_image,
    preprocess_vln_memory_image,
)
from src.train.utils import build_chat_template_prompt

SYSTEM_PROMPT = VLN_SYSTEM_PROMPT
DEFAULT_EVAL_MODEL_PATH = "/workspace/code_dir/a_property/model/Qwen3.5-4B"
TARGET_KEYS = ("success", "spl", "oracle_success", "distance_to_goal", "path_length", "ndtw")
RESULT_FILENAME = "result.jsonl"
RESULT_SUMMARY_FILENAME = "result_summary.json"
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-\d+$")

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.getLogger("imageio.plugins.ffmpeg").setLevel(logging.ERROR)

ATOMIC_ACTION_NAMES = ("stop", "move_forward", "turn_left", "turn_right")
ATOMIC_ACTION_TO_ID = {action_name: action_id for action_id, action_name in enumerate(ATOMIC_ACTION_NAMES)}
ATOMIC_ACTION_ALIASES = {
    "stop": ("stop",),
    "move_forward": ("move_forward", "move forward", "move-forward", "forward"),
    "turn_left": ("turn_left", "turn left", "turn-left", "left"),
    "turn_right": ("turn_right", "turn right", "turn-right", "right"),
}
ATOMIC_ACTION_REGEXES = {
    action_name: tuple(
        re.compile(rf"(?<![0-9a-z_]){re.escape(alias)}(?![0-9a-z_])")
        for alias in aliases
    )
    for action_name, aliases in ATOMIC_ACTION_ALIASES.items()
}


def seed_all(seed=41):
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
    placeholder_images = [f"image_{index}" for index in range(len(images))]
    user_content_template = build_vln_user_content(
        instruction=instruction,
        user_images=placeholder_images,
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


def select_vln_eval_image_indices(
    history_length: int,
    max_memory_images: int,
    memory_pool_window_frames: int,
) -> List[int]:
    last_frame_index = history_length - 1
    if last_frame_index < 0:
        return []
    return build_vln_image_selection(
        current_step=last_frame_index,
        last_frame_index=last_frame_index,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
    )


def preprocess_vln_eval_images(
    rgb_history: Sequence[Image.Image],
    selected_indices: Sequence[int],
) -> List[Image.Image]:
    selected_images = []
    for image_position, frame_index in enumerate(selected_indices):
        raw_image = rgb_history[frame_index]
        is_current_observation = image_position == len(selected_indices) - 1
        if is_current_observation:
            selected_images.append(
                preprocess_vln_current_image(
                    image=raw_image,
                    add_visual_prompt=False,
                )
            )
        else:
            selected_images.append(preprocess_vln_memory_image(raw_image))
    return selected_images


def _strip_generation_wrappers(output: str) -> str:
    output_match = re.search(r"<answer>(.*?)</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    action_text = output_match.group(1) if output_match else output
    action_text = re.sub(r"</?think>", " ", action_text, flags=re.IGNORECASE)
    action_text = re.sub(r"</?answer>", " ", action_text, flags=re.IGNORECASE)
    return " ".join(action_text.split()).strip()


def _normalize_action_candidate(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.lower().strip(" \t\r\n`'\".,;:!?()[]{}"))


def _match_atomic_actions(text: str) -> List[Tuple[int, str]]:
    normalized_text = text.lower()
    matches = []
    for action_name, patterns in ATOMIC_ACTION_REGEXES.items():
        first_match_position = None
        for pattern in patterns:
            match = pattern.search(normalized_text)
            if match is None:
                continue
            if first_match_position is None or match.start() < first_match_position:
                first_match_position = match.start()
        if first_match_position is not None:
            matches.append((first_match_position, action_name))
    matches.sort(key=lambda item: item[0])
    return matches


def parse_atomic_action(output: str):
    action_text = _strip_generation_wrappers(output)
    if not action_text:
        return None

    normalized_candidate = _normalize_action_candidate(action_text)
    if normalized_candidate in ATOMIC_ACTION_TO_ID:
        return ATOMIC_ACTION_TO_ID[normalized_candidate]

    matched_actions = _match_atomic_actions(action_text)
    if not matched_actions:
        return None

    unique_actions = {action_name for _, action_name in matched_actions}
    if len(unique_actions) != 1:
        return None

    _, action_name = matched_actions[0]
    return ATOMIC_ACTION_TO_ID[action_name]


def _build_action_token_prefix_map(action_token_sequences: Sequence[Sequence[int]]) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    prefix_map = {}
    for sequence in action_token_sequences:
        prefix = ()
        for token_id in sequence:
            next_tokens = prefix_map.setdefault(prefix, set())
            next_tokens.add(int(token_id))
            prefix = prefix + (int(token_id),)
        prefix_map.setdefault(prefix, set())
    return {
        prefix: tuple(sorted(token_ids))
        for prefix, token_ids in prefix_map.items()
    }


class AtomicActionPrefixConstraint:
    def __init__(self, tokenizer, eos_token_id):
        self.action_token_sequences = {
            action_name: tuple(
                tokenizer(action_name, add_special_tokens=False)["input_ids"]
            )
            for action_name in ATOMIC_ACTION_NAMES
        }
        invalid_actions = [
            action_name
            for action_name, token_ids in self.action_token_sequences.items()
            if not token_ids
        ]
        if invalid_actions:
            raise ValueError(
                "Failed to tokenize constrained atomic actions: "
                + ", ".join(invalid_actions)
            )

        self.prefix_map = _build_action_token_prefix_map(self.action_token_sequences.values())
        self.complete_sequences = set(self.action_token_sequences.values())
        self.eos_token_id = eos_token_id
        self.prompt_length = None
        self.min_completion_tokens = max(len(token_ids) for token_ids in self.action_token_sequences.values())
        if self.eos_token_id is not None:
            self.min_completion_tokens += 1

    def set_prompt_length(self, prompt_length: int) -> None:
        self.prompt_length = int(prompt_length)

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        del batch_id
        if self.prompt_length is None:
            raise RuntimeError("prompt_length must be set before constrained generation")

        generated_token_ids = tuple(input_ids[self.prompt_length :].tolist())
        allowed_tokens = self.prefix_map.get(generated_token_ids)
        if allowed_tokens:
            return list(allowed_tokens)

        if generated_token_ids in self.complete_sequences:
            if self.eos_token_id is not None:
                return [self.eos_token_id]
            return []

        if self.eos_token_id is not None:
            return [self.eos_token_id]
        return list(self.prefix_map.get((), ()))

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
    merged_path = result_path / RESULT_FILENAME
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
    agent = NaVIDA_Agent(
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
        for episode in progress:
            agent.reset()
            env.current_episode = episode
            obs = env.reset()
            iter_step = 0

            while not env.episode_over:
                info = env.get_metrics()
                action = agent.act(obs, info, env.current_episode.episode_id)

                if early_stop_max_steps > 0 and iter_step >= early_stop_max_steps:
                    action = {"action": 0}

                iter_step += 1
                obs = env.step(action)

            info = env.get_metrics()
            agent.finalize_episode()
            result_row = _result_row(episode.scene_id, episode.episode_id, info)
            result_row["model_generated_actions"] = list(agent.model_generated_actions)
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

class NaVIDA_Agent(Agent):
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
        
        print("Initialize NaVIDA")
        
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

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)

        if lora_path is not None and lora_path!= '':
            print('Loading LoRA weights...')
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print('Merging LoRA weights...')
            self.model = self.model.merge_and_unload()
            print('Model is loaded...')

        self.device = 'cuda'
        self.model.to(self.device)
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("Eval requires a tokenizer to constrain generation to atomic actions")
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        self.atomic_action_constraint = AtomicActionPrefixConstraint(
            tokenizer=self.tokenizer,
            eos_token_id=self.eos_token_id,
        )

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
        print(f"Initialization Complete (attn_implementation={self.attn_implementation})")
        
        self.rgb_history = []
        self.current_images = []
        self.executed_action_history = []
        self.last_returned_action = None
        self.model_generated_actions = []
        self.topdown_frames = []
        self.conversations = []

        self.reset()


    def predict_inference(self):
        texts = [build_chat_template_prompt(self.processor, self.conversations)]

        prompt_inputs = self.processor(
            text=texts,
            images=self.current_images if self.current_images else None,
            return_tensors="pt",
            padding=True,
        )

        prompt_inputs.to(self.device)
        input_token_len = int(prompt_inputs["input_ids"].shape[1])
        self.atomic_action_constraint.set_prompt_length(input_token_len)
        with torch.inference_mode():
            outputs = self.model.generate(
                **prompt_inputs,
                do_sample=False,
                use_cache=True,
                num_return_sequences=1,
                max_new_tokens=self.atomic_action_constraint.min_completion_tokens,
                prefix_allowed_tokens_fn=self.atomic_action_constraint.prefix_allowed_tokens_fn,
            )
        output_ids = outputs
        n_diff_input_output = (prompt_inputs["input_ids"] != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs_text = self.processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs_text = outputs_text[0]
        outputs_text = outputs_text.strip()
        return outputs_text

    def _record_previous_action(self):
        if self.last_returned_action is None:
            return
        self.executed_action_history.append(self.last_returned_action)
        self.last_returned_action = None

    def _select_image_indices(self):
        return select_vln_eval_image_indices(
            history_length=len(self.rgb_history),
            max_memory_images=self.max_memory_images,
            memory_pool_window_frames=self.memory_pool_window_frames,
        )

    def _prepare_selected_images(self, selected_indices):
        return preprocess_vln_eval_images(
            rgb_history=self.rgb_history,
            selected_indices=selected_indices,
        )

    def _predict_action_from_images(self, instruction, selected_images):
        self.current_images = selected_images
        self.conversations = build_eval_messages(
            instruction=instruction,
            images=selected_images,
        )

        navigation = self.predict_inference()
        self.model_generated_actions.append(navigation)
        action_id = parse_atomic_action(navigation)
        return navigation, action_id

    def finalize_episode(self):
        self._record_previous_action()

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
        self.executed_action_history = []
        self.last_returned_action = None
        self.model_generated_actions = []
        self.conversations = []
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        if self.save_topdown and info.get("top_down_map") is not None:
            top_down_frame = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map"], observations["rgb"].shape[0]
            )
            render_frame = np.concatenate((observations["rgb"], top_down_frame), axis=1)
            self.topdown_frames.append(render_frame)

        self._record_previous_action()

        rgb = observations["rgb"]
        self.rgb_history.append(Image.fromarray(rgb.astype('uint8')).convert('RGB'))

        selected_indices = self._select_image_indices()
        selected_images = self._prepare_selected_images(selected_indices)
        navigation, action_id = self._predict_action_from_images(
            instruction=observations["instruction"]["text"],
            selected_images=selected_images,
        )

        if action_id is None:
            print(
                f"[Warning] Failed to parse a valid action from model output on episode {episode_id}: "
                f"{navigation!r}. Defaulting to stop."
            )
            action_id = 0

        self.last_returned_action = action_id

        return {"action": action_id}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--split-id",type=int,required=True,help="chunks ID of evluation")
    parser.add_argument("--model-path",type=str,default=DEFAULT_EVAL_MODEL_PATH,help="location of fully saved model weights")
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
    parser.add_argument("--seed", type=int, default=41, help="random seed for python, numpy, and torch")
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

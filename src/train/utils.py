import importlib
import os
import random
import re
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

try:
    from src.qwen_vl import Qwen3_5Config, Qwen3_5ForConditionalGenerationForPanoVLN
except ModuleNotFoundError:
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from qwen_vl import Qwen3_5Config, Qwen3_5ForConditionalGenerationForPanoVLN


DEFAULT_TRAINABLE_MODULES = {
    "visual": True,
    "visual_merger": True,
    "language_model": True,
    "erp_fourier_linear_adapter": True,
    "da2_mlp": True,
}
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank0_print(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def _get_dtype(dtype_str: Optional[str]):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    return None


def set_model(cfg, model):
    trainable_modules = dict(DEFAULT_TRAINABLE_MODULES)
    if cfg.model.trainable_modules:
        trainable_modules.update(cfg.model.trainable_modules)

    for param in model.parameters():
        param.requires_grad = False

    visual_model = getattr(model, "visual", None)
    if visual_model is None and hasattr(model, "model"):
        visual_model = getattr(model.model, "visual", None)

    language_model = getattr(model, "language_model", None)
    if language_model is None and hasattr(model, "model"):
        language_model = getattr(model.model, "language_model", None)

    named_modules = {
        "visual": visual_model,
        "visual_merger": (
            getattr(visual_model, "merger", None)
            if visual_model is not None else None
        ),
        "erp_fourier_linear_adapter": getattr(model, "erp_fourier_linear_adapter", None),
        "da2_mlp": getattr(model, "da2_mlp", None),
        "language_model": language_model,
    }

    for module_name, module in named_modules.items():
        if not trainable_modules.get(module_name) or module is None:
            continue
        if hasattr(module, "enabled") and not getattr(module, "enabled"):
            continue
        for _, param in module.named_parameters():
            param.requires_grad = True

    if trainable_modules.get("language_model"):
        if hasattr(model, "lm_head"):
            for _, param in model.lm_head.named_parameters():
                param.requires_grad = True


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name} | requires_grad={param.requires_grad}")


def _load_model_config(cfg):
    config = Qwen3_5Config.from_pretrained(
        cfg.model.name_or_path,
        cache_dir=cfg.model.cache_dir,
    )
    erp_fourier_linear_fields = (
        "erp_fourier_linear_enabled",
        "erp_fourier_linear_alpha_value",
        "erp_fourier_linear_apply_to_current_only",
    )
    erp_crop_fields = (
        "erp_top_crop_degrees",
        "erp_bottom_crop_degrees",
    )
    da2_fields = (
        "da2_enabled",
        "da2_source_path",
        "da2_model_path",
        "da2_alpha_value",
        "da2_feature_source",
        "da2_injection_stage",
        "da2_sampling_mode",
        "da2_force_fp32",
    )

    def apply_module_fields(enabled: bool, field_names: tuple[str, ...]) -> None:
        if enabled:
            for field_name in field_names:
                setattr(config, field_name, getattr(cfg.model, field_name))
            return
        for field_name in field_names:
            if hasattr(config, field_name):
                delattr(config, field_name)

    def apply_vision_module_fields(enabled: bool, field_names: tuple[str, ...]) -> None:
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            return
        if enabled:
            for field_name in field_names:
                value = getattr(cfg.model, field_name)
                if value is not None:
                    setattr(vision_config, field_name, value)
            return
        for field_name in field_names:
            if hasattr(vision_config, field_name):
                delattr(vision_config, field_name)

    def apply_module_fields_preserve_checkpoint(enabled: bool, field_names: tuple[str, ...]) -> None:
        checkpoint_enabled = bool(getattr(config, field_names[0], False))
        if enabled:
            apply_module_fields(True, field_names)
            return
        if checkpoint_enabled:
            for field_name in field_names:
                if not hasattr(config, field_name):
                    setattr(config, field_name, getattr(cfg.model, field_name))
            return
        apply_module_fields(False, field_names)

    apply_vision_module_fields(bool(cfg.model.erp_fourier_linear_enabled), erp_fourier_linear_fields)
    for field_name in erp_crop_fields:
        setattr(config, field_name, getattr(cfg.model, field_name))
    apply_module_fields_preserve_checkpoint(bool(cfg.model.da2_enabled), da2_fields)
    return config


def load_model(cfg):
    torch_dtype = _get_dtype(cfg.model.torch_dtype)
    model = Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(
        cfg.model.name_or_path,
        config=_load_model_config(cfg),
        torch_dtype=torch_dtype,
        cache_dir=cfg.model.cache_dir,
        attn_implementation=cfg.model.attn_implementation,
    )

    if cfg.training.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.config.use_cache = False
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.use_cache = False
    model.accepts_loss_kwargs = False
    return model


def checkpoint_has_da2_encoder_weights(pretrained_model_name_or_path: str) -> bool:
    return bool(
        Qwen3_5ForConditionalGenerationForPanoVLN._checkpoint_has_any_weights(
            pretrained_model_name_or_path,
            ("da2.dino.cls_token",),
        )
    )


def sync_model_special_tokens(model, tokenizer):
    special_token_fields = (
        ("eos_token_id", tokenizer.eos_token_id),
        ("pad_token_id", tokenizer.pad_token_id),
        ("bos_token_id", tokenizer.bos_token_id),
    )

    for field_name, token_id in special_token_fields:
        if token_id is None:
            continue

        setattr(model.config, field_name, token_id)

        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            setattr(text_config, field_name, token_id)

        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, field_name, token_id)


def load_processor_and_tokenizer(cfg):
    processor = AutoProcessor.from_pretrained(
        cfg.model.name_or_path,
        use_fast=True,
        trust_remote_code=cfg.model.trust_remote_code,
        cache_dir=cfg.model.cache_dir,
    )
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path,
            trust_remote_code=cfg.model.trust_remote_code,
            cache_dir=cfg.model.cache_dir,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor, tokenizer


def load_wandb_module(required: bool = False):
    try:
        import wandb as wandb_module
        if hasattr(wandb_module, "init") or hasattr(wandb_module, "finish"):
            return wandb_module
    except Exception:
        pass

    cwd = os.path.abspath(os.getcwd())
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("wandb", None)
        sys.path[:] = [
            path for path in sys.path
            if os.path.abspath(path or cwd) != cwd
        ]
        wandb_module = importlib.import_module("wandb")
    except Exception:
        wandb_module = None
    finally:
        sys.path[:] = original_sys_path

    if required and (wandb_module is None or not hasattr(wandb_module, "init")):
        raise ImportError(
            "report_to includes 'wandb', but the real wandb package could not be imported. "
            "A local ./wandb directory may be shadowing it."
        )
    return wandb_module


def init_wandb(wandb_cfg, training_args, rank):
    if wandb_cfg is None:
        return
    report_to = training_args.report_to
    if report_to is None:
        return
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in report_to:
        return
    if rank != 0:
        return

    wandb = load_wandb_module(required=True)

    if is_dataclass(wandb_cfg):
        cfg_dict = asdict(wandb_cfg)
    elif isinstance(wandb_cfg, dict):
        cfg_dict = dict(wandb_cfg)
    else:
        cfg_dict = {
            k: getattr(wandb_cfg, k)
            for k in ("project", "entity", "name", "tags", "mode", "group", "notes", "id", "resume")
            if hasattr(wandb_cfg, k)
        }

    init_kwargs = {k: v for k, v in cfg_dict.items() if v is not None}
    if not init_kwargs.get("name") and getattr(training_args, "run_name", None):
        init_kwargs["name"] = training_args.run_name

    wandb.init(**init_kwargs)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def build_prompt_and_target(
    messages: List[Dict[str, Any]],
    prompt_format: str,
    processor=None,
    require_target: bool = True,
) -> Dict[str, str]:
    if not messages:
        raise ValueError("messages is empty")

    target_text = ""
    if messages[-1].get("role") == "assistant":
        target_msg = messages[-1]
        target_text = _content_to_text(target_msg.get("content", ""))
        prompt_messages = messages[:-1]
    elif require_target:
        raise ValueError("The last message must be from the assistant")
    else:
        prompt_messages = messages

    if prompt_format == "chat_template":
        if processor is None or not hasattr(processor, "apply_chat_template"):
            raise ValueError("chat_template prompt_format requires processor.apply_chat_template")
        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"prompt": prompt_text, "target": target_text}

    chunks = []
    for message in prompt_messages:
        role = message.get("role", "user")
        text = _content_to_text(message.get("content", ""))
        chunks.append(f"<|{role}|>\n{text}\n")
    chunks.append("<|assistant|>\n")
    return {"prompt": "".join(chunks), "target": target_text}


def preprocess_logits_for_metrics(logits, labels):
    del labels
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def _normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


ACTION_PATTERN_VARIANTS = {
    "forward": ("forward", "move_forward", "move forward", "move-forward"),
    "left": ("left", "turn_left", "turn left", "turn-left"),
    "right": ("right", "turn_right", "turn right", "turn-right"),
    "stop": ("stop",),
    "move_forward": ("forward", "move_forward", "move forward", "move-forward"),
    "turn_left": ("left", "turn_left", "turn left", "turn-left"),
    "turn_right": ("right", "turn_right", "turn right", "turn-right"),
}
ACTION_METRIC_NAMES = {
    "forward": "forward",
    "left": "left",
    "right": "right",
    "move_forward": "forward",
    "turn_left": "left",
    "turn_right": "right",
}


def _action_metric_name(action: str) -> str:
    metric_action = ACTION_METRIC_NAMES.get(action, action)
    return re.sub(r"[^0-9a-z]+", "_", metric_action.lower()).strip("_")


def _build_action_patterns(action_vocab: List[str]) -> List[tuple]:
    patterns = []
    for action in action_vocab:
        variants = ACTION_PATTERN_VARIANTS.get(
            action,
            (action, action.replace("_", " "), action.replace("_", "-")),
        )
        variant_patterns = []
        for variant in variants:
            normalized_variant = _normalize_action_text(variant)
            escaped_variant = re.escape(normalized_variant)
            variant_patterns.append(
                r"(?<![0-9a-z_])" + escaped_variant + r"(?![0-9a-z_])"
            )
        pattern = r"(?:" + "|".join(variant_patterns) + r")"
        patterns.append((action, re.compile(pattern)))
    return patterns


def _extract_action_sequence(
    text: str,
    patterns: List[tuple],
    max_actions: Optional[int] = None,
) -> List[str]:
    normalized = _normalize_action_text(text)
    matches = []
    for action, pattern in patterns:
        for match in pattern.finditer(normalized):
            matches.append((match.start(), match.end(), action))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    actions = []
    last_end = -1
    for start, end, action in matches:
        if start < last_end:
            continue
        actions.append(action)
        last_end = end
        if max_actions is not None and len(actions) >= max_actions:
            break
    return actions


def _trim_padded_stop_actions(actions: List[str]) -> List[str]:
    if "stop" not in actions:
        return actions
    return actions[:actions.index("stop") + 1]


def _build_action_weights(
    action_vocab: List[str],
    f1_action_weight: Optional[List[float]],
) -> Dict[str, float]:
    if f1_action_weight is None:
        return {action: 1.0 for action in action_vocab}
    if len(f1_action_weight) != len(action_vocab):
        raise ValueError("f1_action_weight length must match action_vocab length")

    action_weights = {}
    for action, weight in zip(action_vocab, f1_action_weight):
        weight = float(weight)
        if weight < 0:
            raise ValueError("f1_action_weight values must be >= 0")
        action_weights[action] = weight

    if sum(action_weights.values()) <= 0:
        raise ValueError("f1_action_weight must contain at least one positive weight")

    return action_weights


def _safe_precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return 0.0 if denom == 0 else float(tp / denom)


def _safe_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return 0.0 if denom == 0 else float(tp / denom)


def _safe_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom == 0 else float(2 * precision * recall / denom)


def build_action_accuracy(
    tokenizer,
    action_vocab: Optional[List[str]] = None,
    f1_action_weight: Optional[List[float]] = None,
):
    if action_vocab is None:
        action_vocab = ["stop", "forward", "left", "right"]

    patterns = _build_action_patterns(action_vocab)
    action_weights = _build_action_weights(action_vocab, f1_action_weight)
    action_metric_names = {
        action: f"{_action_metric_name(action)}_accuracy"
        for action in action_vocab
    }
    action_f1_metric_names = {
        action: f"{_action_metric_name(action)}_f1"
        for action in action_vocab
    }

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    def _sanitize(array: Any):
        array = np.asarray(array).astype(np.int64, copy=False)
        array = np.where(array < 0, pad_id, array)
        array = np.where(array >= vocab_size, pad_id, array)
        return array

    def _shift_for_causal_lm(preds):
        shifted = np.full_like(preds, pad_id)
        shifted[:, 1:] = preds[:, :-1]
        return shifted

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)
        labels = np.asarray(labels)

        if preds.ndim == labels.ndim + 1:
            preds = np.argmax(preds, axis=-1)

        preds = _sanitize(preds)
        labels = labels.astype(np.int64, copy=False)

        shifted_preds = _shift_for_causal_lm(preds)
        pred_ids = np.where(labels != -100, shifted_preds, pad_id)
        label_ids = np.where(labels != -100, labels, pad_id)

        pred_ids = _sanitize(pred_ids)
        label_ids = _sanitize(label_ids)

        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        total = 0
        correct = 0
        sequence_total = 0
        sequence_correct = 0
        position_total = [0, 0, 0, 0]
        position_correct = [0, 0, 0, 0]
        per_action_total = {action: 0 for action in action_vocab}
        per_action_correct = {action: 0 for action in action_vocab}
        per_action_tp = {action: 0 for action in action_vocab}
        per_action_fp = {action: 0 for action in action_vocab}
        per_action_fn = {action: 0 for action in action_vocab}

        for pred_text, label_text in zip(pred_texts, label_texts):
            label_actions = _trim_padded_stop_actions(
                _extract_action_sequence(label_text, patterns)
            )

            if not label_actions:
                continue

            pred_actions = _extract_action_sequence(
                pred_text,
                patterns,
                max_actions=len(label_actions),
            )
            sequence_total += 1
            if pred_actions == label_actions:
                sequence_correct += 1

            for action_index, label_action in enumerate(label_actions):
                pred_action = (
                    pred_actions[action_index]
                    if action_index < len(pred_actions)
                    else None
                )
                action_correct = pred_action == label_action

                if action_index < len(position_total):
                    position_total[action_index] += 1
                    if action_correct:
                        position_correct[action_index] += 1

                total += 1
                per_action_total[label_action] += 1
                if action_correct:
                    correct += 1
                    per_action_correct[label_action] += 1
                    per_action_tp[label_action] += 1
                    continue

                per_action_fn[label_action] += 1
                if pred_action in per_action_fp:
                    per_action_fp[pred_action] += 1

        metrics = {
            "action_accuracy": 0.0 if total == 0 else float(correct / total),
            "sequence_accuracy": (
                0.0
                if sequence_total == 0
                else float(sequence_correct / sequence_total)
            ),
        }

        for position_index, position_action_total in enumerate(position_total):
            metrics[f"position_{position_index + 1}_accuracy"] = (
                0.0
                if position_action_total == 0
                else float(position_correct[position_index] / position_action_total)
            )

        weighted_f1_num = 0.0
        weighted_f1_den = 0.0
        for action in action_vocab:
            action_total = per_action_total[action]
            metrics[action_metric_names[action]] = (
                0.0
                if action_total == 0
                else float(per_action_correct[action] / action_total)
            )
            precision = _safe_precision(per_action_tp[action], per_action_fp[action])
            recall = _safe_recall(per_action_tp[action], per_action_fn[action])
            f1 = _safe_f1(precision, recall)
            metrics[action_f1_metric_names[action]] = f1

            if action_total > 0 and action_weights[action] > 0:
                weighted_f1_num += action_weights[action] * f1
                weighted_f1_den += action_weights[action]

        metrics["weighted_action_f1"] = (
            0.0 if weighted_f1_den == 0 else float(weighted_f1_num / weighted_f1_den)
        )

        return metrics

    return compute_metrics

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name_or_path: str
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    cache_dir: Optional[str] = None
    image_token: str = "<|image_pad|>"
    model_max_length: Optional[int] = 163840
    trainable_modules: Optional[Dict[str, bool]] = None
    erp_top_crop_degrees: float = 0.0
    erp_bottom_crop_degrees: float = 0.0
    erp_fourier_linear_enabled: bool = False
    erp_fourier_linear_alpha_value: float = 0.01
    erp_fourier_linear_apply_to_current_only: bool = True
    da2_enabled: bool = True
    da2_source_path: str = "bundled"
    da2_model_path: str = "/workspace/data1/model/DA-2"
    da2_alpha_value: float = 0.05
    da2_feature_source: str = "decoder_multiscale"
    da2_injection_stage: str = "post_merger"
    da2_sampling_mode: str = "grouping"
    da2_force_fp32: bool = False


@dataclass
class DataConfig:
    train_jsonl: str = "/workspace/data1/dataset/Panoworld/train_outdoor.jsonl"
    eval_jsonl: Optional[str] = None
    train_image_root: Optional[str] = "/workspace/data1/dataset/Panoworld"
    eval_image_root: Optional[str] = None
    train_max_samples: Optional[int] = None
    eval_max_samples: Optional[int] = None
    shuffle: bool = True
    eval_shuffle: bool = False
    prompt_format: str = "chat_template"
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = "/workspace/code/VLN/src/panoworld/config/system_prompts/erp_multimodal_prompts.txt"
    auto_insert_media_placeholders: bool = True
    image_processor: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    output_dir: str = "/workspace/data1/model/panoworld/panovln_da2_1e-6"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-6
    language_model_lr: Optional[float] = 1.0e-6
    visual_lr: Optional[float] = 1.0e-6
    visual_merger_lr: Optional[float] = 1.0e-6
    erp_fourier_linear_lr: Optional[float] = 1.0e-6
    da2_mlp_lr: Optional[float] = 1.0e-6
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    logging_steps: int = 5
    save_steps: int = 300
    eval_steps: int = 300
    max_steps: int = -1
    eval_strategy: str = "no"
    save_strategy: str = "steps"
    save_model_at_end: bool = True
    max_shard_size: str = "5GB"
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    save_total_limit: Optional[int] = 2
    warmup_steps: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    optim: Optional[str] = "adamw_torch"
    report_to: Optional[List[str]] = None
    run_name: Optional[str] = None
    seed: int = 42
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 6
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    deepspeed: Optional[str] = "/workspace/code/VLN/scripts/zero3.json"


@dataclass
class WandbConfig:
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: Optional[str] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    id: Optional[str] = None
    resume: Optional[str] = None


@dataclass
class RunConfig:
    do_train: bool = True
    do_eval: bool = False
    resume_from_checkpoint: Optional[Any] = False


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    run: RunConfig
    wandb: Optional[WandbConfig] = None
    config_dir: str = "."


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None or path == "":
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _load_system_prompt(data_cfg: DataConfig, config_dir: str) -> None:
    if data_cfg.system_prompt:
        return
    if not data_cfg.system_prompt_path:
        return
    prompt_path = _resolve_path(data_cfg.system_prompt_path, config_dir)
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as handle:
            data_cfg.system_prompt = handle.read().strip()


def _resolve_config_paths(cfg: TrainConfig) -> TrainConfig:
    config_dir = cfg.config_dir
    cfg.model.name_or_path = _resolve_path(cfg.model.name_or_path, config_dir)
    cfg.model.cache_dir = _resolve_path(cfg.model.cache_dir, config_dir)
    if str(cfg.model.da2_source_path).strip().lower() != "bundled":
        cfg.model.da2_source_path = _resolve_path(cfg.model.da2_source_path, config_dir)
    cfg.model.da2_model_path = _resolve_path(cfg.model.da2_model_path, config_dir)

    cfg.data.train_jsonl = _resolve_path(cfg.data.train_jsonl, config_dir)
    cfg.data.eval_jsonl = _resolve_path(cfg.data.eval_jsonl, config_dir)
    cfg.data.train_image_root = _resolve_path(cfg.data.train_image_root, config_dir)
    cfg.data.eval_image_root = _resolve_path(cfg.data.eval_image_root, config_dir)
    _load_system_prompt(cfg.data, config_dir)

    cfg.training.output_dir = _resolve_path(cfg.training.output_dir, config_dir)
    cfg.training.deepspeed = _resolve_path(cfg.training.deepspeed, config_dir)
    return cfg


def load_config(path: str) -> TrainConfig:
    config_path = os.path.abspath(path)
    config_dir = os.path.dirname(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = TrainConfig(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        run=RunConfig(**raw.get("run", {})),
        wandb=WandbConfig(**raw["wandb"]) if raw.get("wandb") else None,
        config_dir=config_dir,
    )
    return _resolve_config_paths(cfg)

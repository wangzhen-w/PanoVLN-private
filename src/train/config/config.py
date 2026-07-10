from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name_or_path: str
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    cache_dir: Optional[str] = None
    image_token: str = "<image>"
    model_max_length: Optional[int] = None
    trainable_modules: Optional[Dict[str, bool]] = None
    erp_top_crop_degrees: float = 20.0
    erp_bottom_crop_degrees: float = 20.0
    erp_fourier_linear_enabled: bool = False
    erp_fourier_linear_alpha_value: float = 0.01
    erp_fourier_linear_apply_to_current_only: bool = True
    panovggt_enabled: bool = False
    panovggt_checkpoint_path: str = "/workspace/code/a_property/model/PanoVGGT/model.pt"
    panovggt_alpha_value: float = 0.1
    panovggt_feature_source: str = "aggregator"
    panovggt_injection_stage: str = "post_merger"
    panovggt_sampling_mode: str = "grouping"
    panovggt_force_fp32: bool = False


@dataclass
class PanoWorldDataConfig:
    enabled: bool = False
    jsonl: Optional[str] = None
    image_root: Optional[str] = None
    keep_ratio: float = 0.0
    mixing_strategy: str = "sample"
    max_samples: Optional[int] = None
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = (
        "/workspace/code/VLN/src/panoworld/config/system_prompts/erp_multimodal_prompts.txt"
    )
    top_crop_degrees: float = 0.0
    bottom_crop_degrees: float = 0.0
    auto_insert_media_placeholders: bool = True


@dataclass
class DataConfig:
    train_jsonl: str
    eval_jsonl: Optional[str] = None
    train_image_root: Optional[str] = None
    eval_image_root: Optional[str] = None
    train_max_samples: Optional[int] = None
    eval_max_samples: Optional[int] = None
    shuffle: bool = True
    action_vocab: Optional[List[str]] = None
    f1_action_weight: Optional[List[float]] = None
    prompt_format: str = "chat_template"
    panoworld: PanoWorldDataConfig = field(default_factory=PanoWorldDataConfig)


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    language_model_lr: Optional[float] = None
    visual_lr: Optional[float] = None
    visual_merger_lr: Optional[float] = None
    erp_fourier_linear_lr: Optional[float] = None
    panovggt_mlp_lr: Optional[float] = None
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_steps: int = -1
    eval_strategy: str = "no"
    save_strategy: str = "no"
    save_model_at_end: bool = True
    max_shard_size: str = "5GB"
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    save_total_limit: Optional[int] = None
    warmup_steps: float = 0.0
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    optim: Optional[str] = None
    report_to: Optional[List[str]] = None
    run_name: Optional[str] = None
    seed: int = 42
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    deepspeed: Optional[str] = None


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
    do_eval: bool = False
    resume_from_checkpoint: bool = False


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    run: RunConfig
    wandb: Optional[WandbConfig] = None


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model = raw.get("model", {})
    data = raw.get("data", {})
    training = raw.get("training", {})
    run = raw.get("run", {})
    wandb = raw.get("wandb", None)
    if "panoworld" in data:
        data = dict(data)
        panoworld = data.get("panoworld") or {}
        data["panoworld"] = PanoWorldDataConfig(**panoworld)

    return TrainConfig(
        model=ModelConfig(**model),
        data=DataConfig(**data),
        training=TrainingConfig(**training),
        run=RunConfig(**run),
        wandb=WandbConfig(**wandb) if wandb else None,
    )

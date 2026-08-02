import argparse
import os
import shutil
from typing import Optional

import yaml
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from config.config import load_config
from data import PanoworldSupervisedDataset

try:
    from src.train.data.collator import MultiModalDataCollator
    from src.train.utils import (
        init_wandb,
        load_model,
        load_processor_and_tokenizer,
        load_wandb_module,
        print_model_parameters,
        rank0_print,
        set_model,
        set_seed,
        sync_model_special_tokens,
    )
except ModuleNotFoundError:
    from train.data.collator import MultiModalDataCollator
    from train.utils import (
        init_wandb,
        load_model,
        load_processor_and_tokenizer,
        load_wandb_module,
        print_model_parameters,
        rank0_print,
        set_model,
        set_seed,
        sync_model_special_tokens,
    )


RANK = int(os.environ.get("RANK", "0"))


class PanoWorldSFTTrainer(Trainer):
    def get_decay_parameter_names(self, model):
        return super().get_decay_parameter_names(model)

    def __init__(self, *args, module_learning_rates=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_learning_rates = {
            name: float(lr)
            for name, lr in (module_learning_rates or {}).items()
            if lr is not None
        }

    @staticmethod
    def _name_has_module(name: str, module_name: str) -> bool:
        return name == module_name or name.startswith(f"{module_name}.") or f".{module_name}." in name

    def _module_lr_key_for_parameter(self, name: str) -> Optional[str]:
        if name.startswith("visual.merger.") or ".visual.merger." in name:
            return "visual_merger"
        if self._name_has_module(name, "panovggt_mlp"):
            return "panovggt_mlp"
        if self._name_has_module(name, "visual"):
            return "visual"
        if (
            self._name_has_module(name, "language_model")
            or self._name_has_module(name, "model")
            or self._name_has_module(name, "lm_head")
        ):
            return "language_model"
        return None

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = set(self.get_decay_parameter_names(opt_model))
        grouped_parameters = {}

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue

            module_key = self._module_lr_key_for_parameter(name)
            lr = self.module_learning_rates.get(module_key, self.args.learning_rate)
            weight_decay = self.args.weight_decay if name in decay_parameters else 0.0
            group_key = (module_key or "base", float(lr), float(weight_decay))
            if group_key not in grouped_parameters:
                grouped_parameters[group_key] = {
                    "params": [],
                    "weight_decay": weight_decay,
                    "lr": lr,
                }
            grouped_parameters[group_key]["params"].append(param)

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(list(grouped_parameters.values()), **optimizer_kwargs)
        return self.optimizer


def _config_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def apply_config_overrides(cfg, overrides) -> None:
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Config override must use key=value format, got: {override}")
        key, raw_value = override.split("=", 1)
        target = cfg
        parts = key.split(".")
        if len(parts) < 2:
            raise ValueError(f"Config override key must be dotted, got: {key}")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"Unknown config section in override: {key}")
            target = getattr(target, part)
        field_name = parts[-1]
        if not hasattr(target, field_name):
            raise ValueError(f"Unknown config field in override: {key}")
        current_value = getattr(target, field_name)
        parsed_value = yaml.safe_load(raw_value)
        if (
            isinstance(current_value, str)
            and not isinstance(parsed_value, str)
            and str(raw_value).strip().lower() not in {"null", "none", "~"}
        ):
            parsed_value = raw_value
        setattr(target, field_name, parsed_value)


def _resolve_resume_checkpoint(resume_from_checkpoint, output_dir: str):
    if resume_from_checkpoint in (None, False, "false", "False", "0", 0):
        return None
    if isinstance(resume_from_checkpoint, str):
        value = resume_from_checkpoint.strip()
        if not value:
            return None
        if value.lower() in {"true", "auto", "1", "yes"}:
            return get_last_checkpoint(output_dir)
        return value
    if resume_from_checkpoint:
        return get_last_checkpoint(output_dir)
    return None


def _validate_required_paths(cfg) -> None:
    required_paths = {
        "model.name_or_path": cfg.model.name_or_path,
    }
    if bool(cfg.run.do_train):
        required_paths["data.train_jsonl"] = cfg.data.train_jsonl
    if bool(cfg.run.do_eval):
        if not cfg.data.eval_jsonl:
            raise ValueError("data.eval_jsonl is required when run.do_eval=true")
        required_paths["data.eval_jsonl"] = cfg.data.eval_jsonl
    if cfg.model.panovggt_enabled:
        required_paths["model.panovggt_checkpoint_path"] = cfg.model.panovggt_checkpoint_path
    if cfg.training.deepspeed:
        required_paths["training.deepspeed"] = cfg.training.deepspeed

    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if path and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n" + "\n".join(missing))
    os.makedirs(cfg.training.output_dir, exist_ok=True)


def print_training_config(cfg) -> None:
    rank0_print(RANK, "===== PanoWorld config =====")
    rank0_print(RANK, f"model: {cfg.model.name_or_path}")
    rank0_print(RANK, f"train_jsonl: {cfg.data.train_jsonl}")
    rank0_print(RANK, f"train_image_root: {cfg.data.train_image_root}")
    rank0_print(RANK, f"output_dir: {cfg.training.output_dir}")
    rank0_print(RANK, "trainable_modules:")
    for name, enabled in (cfg.model.trainable_modules or {}).items():
        rank0_print(RANK, f"  {name}: {_config_value(enabled)}")
    rank0_print(RANK, f"erp_top_crop_degrees: {_config_value(cfg.model.erp_top_crop_degrees)}")
    rank0_print(RANK, f"erp_bottom_crop_degrees: {_config_value(cfg.model.erp_bottom_crop_degrees)}")
    rank0_print(RANK, f"panovggt_enabled: {_config_value(cfg.model.panovggt_enabled)}")
    rank0_print(RANK, f"panovggt_alpha_value: {_config_value(cfg.model.panovggt_alpha_value)}")
    rank0_print(RANK, f"panovggt_feature_source: {_config_value(cfg.model.panovggt_feature_source)}")
    rank0_print(RANK, f"panovggt_injection_stage: {_config_value(cfg.model.panovggt_injection_stage)}")
    rank0_print(RANK, f"panovggt_sampling_mode: {_config_value(cfg.model.panovggt_sampling_mode)}")
    rank0_print(RANK, f"per_device_train_batch_size: {cfg.training.per_device_train_batch_size}")
    rank0_print(RANK, f"gradient_accumulation_steps: {cfg.training.gradient_accumulation_steps}")
    rank0_print(RANK, f"learning_rate: {cfg.training.learning_rate}")
    rank0_print(RANK, f"language_model_lr: {_config_value(cfg.training.language_model_lr)}")
    rank0_print(RANK, f"visual_lr: {_config_value(cfg.training.visual_lr)}")
    rank0_print(RANK, f"visual_merger_lr: {_config_value(cfg.training.visual_merger_lr)}")
    rank0_print(RANK, f"panovggt_mlp_lr: {_config_value(cfg.training.panovggt_mlp_lr)}")
    rank0_print(RANK, "============================")


def copy_chat_template_files(source_dir: str, output_dir: str) -> None:
    for template_name in ("chat_template.json", "chat_template.jinja"):
        source_path = os.path.join(source_dir, template_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(output_dir, template_name))


def safe_save_model_for_hf_trainer(
    trainer: Trainer,
    output_dir: str,
    max_shard_size: str = "5GB",
) -> None:
    if trainer.accelerator is None:
        return

    trainer.accelerator.wait_for_everyone()
    state_dict_model = trainer.model_wrapped if trainer.is_deepspeed_enabled else trainer.model
    state_dict = trainer.accelerator.get_state_dict(state_dict_model)
    model_to_save = trainer.accelerator.unwrap_model(trainer.model)

    if trainer.args.should_save:
        safe_serialization = getattr(trainer.args, "save_safetensors", True)
        model_to_save.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )

    trainer.accelerator.wait_for_everyone()


def _build_dataset(cfg, processor, tokenizer, *, split: str, panovggt_enabled: bool):
    is_train = split == "train"
    jsonl_path = cfg.data.train_jsonl if is_train else cfg.data.eval_jsonl
    image_root = cfg.data.train_image_root if is_train else (cfg.data.eval_image_root or cfg.data.train_image_root)
    if not jsonl_path:
        return None
    return PanoworldSupervisedDataset(
        jsonl_path=jsonl_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root=image_root,
        image_token=cfg.model.image_token,
        model_max_length=cfg.model.model_max_length,
        erp_top_crop_degrees=cfg.model.erp_top_crop_degrees,
        erp_bottom_crop_degrees=cfg.model.erp_bottom_crop_degrees,
        panovggt_enabled=panovggt_enabled,
        max_samples=cfg.data.train_max_samples if is_train else cfg.data.eval_max_samples,
        shuffle=cfg.data.shuffle if is_train else cfg.data.eval_shuffle,
        prompt_format=cfg.data.prompt_format,
        system_prompt=cfg.data.system_prompt,
        auto_insert_media_placeholders=cfg.data.auto_insert_media_placeholders,
        image_processor_cfg=cfg.data.image_processor,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values, e.g. --set training.max_steps=30",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_config_overrides(cfg, args.set)
    output_dir_override = os.environ.get("OUTPUT_DIR")
    if output_dir_override:
        cfg.training.output_dir = output_dir_override

    _validate_required_paths(cfg)
    if RANK == 0:
        print_training_config(cfg)
    set_seed(cfg.training.seed)

    processor, tokenizer = load_processor_and_tokenizer(cfg)
    model = load_model(cfg)
    sync_model_special_tokens(model, tokenizer)

    model_config = model.config
    effective_panovggt_enabled = bool(
        getattr(model_config, "panovggt_enabled", cfg.model.panovggt_enabled)
    )

    train_dataset = (
        _build_dataset(
            cfg,
            processor,
            tokenizer,
            split="train",
            panovggt_enabled=effective_panovggt_enabled,
        )
        if cfg.run.do_train else None
    )
    if cfg.run.do_train and train_dataset is None:
        raise ValueError("run.do_train=true requires data.train_jsonl")

    eval_dataset = (
        _build_dataset(
            cfg,
            processor,
            tokenizer,
            split="eval",
            panovggt_enabled=effective_panovggt_enabled,
        )
        if cfg.run.do_eval else None
    )

    if cfg.run.do_train:
        set_model(cfg, model)
    else:
        for param in model.parameters():
            param.requires_grad = False

    load_best_model_at_end = bool(
        cfg.run.do_train
        and cfg.training.load_best_model_at_end
        and eval_dataset is not None
    )

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        seed=cfg.training.seed,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        max_steps=cfg.training.max_steps,
        eval_strategy=(
            cfg.training.eval_strategy
            if cfg.run.do_train and eval_dataset is not None else "no"
        ),
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=(
            cfg.training.metric_for_best_model if load_best_model_at_end else None
        ),
        greater_is_better=(
            cfg.training.greater_is_better if load_best_model_at_end else None
        ),
        save_total_limit=cfg.training.save_total_limit,
        warmup_steps=cfg.training.warmup_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
        remove_unused_columns=cfg.training.remove_unused_columns,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        max_grad_norm=cfg.training.max_grad_norm,
        deepspeed=cfg.training.deepspeed,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if RANK == 0:
        rank0_print(RANK, "===== Effective model config =====")
        rank0_print(RANK, f"panovggt_enabled: {_config_value(effective_panovggt_enabled)}")
        rank0_print(RANK, f"panovggt_alpha_value: {_config_value(getattr(model_config, 'panovggt_alpha_value', None))}")
        rank0_print(RANK, f"panovggt_feature_source: {_config_value(getattr(model_config, 'panovggt_feature_source', None))}")
        rank0_print(RANK, f"panovggt_injection_stage: {_config_value(getattr(model_config, 'panovggt_injection_stage', None))}")
        rank0_print(RANK, f"panovggt_sampling_mode: {_config_value(getattr(model_config, 'panovggt_sampling_mode', None))}")
        rank0_print(RANK, "==================================")
        print_model_parameters(model)

    init_wandb(cfg.wandb, training_args, RANK)

    trainer = PanoWorldSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultiModalDataCollator(tokenizer),
        processing_class=tokenizer,
        module_learning_rates={
            "language_model": cfg.training.language_model_lr,
            "visual": cfg.training.visual_lr,
            "visual_merger": cfg.training.visual_merger_lr,
            "panovggt_mlp": cfg.training.panovggt_mlp_lr,
        },
    )

    if not cfg.run.do_train:
        if eval_dataset is not None:
            metrics = trainer.evaluate()
            if RANK == 0:
                rank0_print(RANK, f"eval metrics: {metrics}")
        return

    rank0_print(RANK, "calling trainer.train()")
    resume_checkpoint = _resolve_resume_checkpoint(
        cfg.run.resume_from_checkpoint,
        training_args.output_dir,
    )
    if resume_checkpoint:
        rank0_print(RANK, f"resuming from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    trainer.save_state()

    if cfg.training.save_model_at_end:
        copy_chat_template_files(
            source_dir=cfg.model.name_or_path,
            output_dir=training_args.output_dir,
        )
        model.config.use_cache = True
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config.use_cache = True
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            max_shard_size=cfg.training.max_shard_size,
        )
        if RANK == 0:
            processor.save_pretrained(cfg.training.output_dir)
            tokenizer.save_pretrained(cfg.training.output_dir)
    else:
        rank0_print(RANK, "save_model_at_end=false, skipping final model/processor/tokenizer save")

    if RANK == 0:
        wandb = load_wandb_module(required=False)
        if wandb is not None and hasattr(wandb, "finish"):
            wandb.finish()


if __name__ == "__main__":
    main()

import argparse
import os
import shutil

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import yaml

from config.config import load_config
from data.collator import MultiModalDataCollator
from data.data import SupervisedDataset
from data.sampler import TraceSampler
from utils import (
    build_action_accuracy,
    init_wandb,
    load_model,
    load_processor_and_tokenizer,
    load_wandb_module,
    preprocess_logits_for_metrics,
    print_model_parameters,
    rank0_print,
    set_model,
    set_seed,
    sync_model_special_tokens,
)

RANK = int(os.environ.get("RANK", "0"))


class PanoVLNTrainer(Trainer):
    RAW_ALPHA_NO_DECAY_SUFFIXES = (
    )
    MODULE_LR_KEYS = (
        "language_model",
        "visual",
        "visual_merger",
        "erp_position_mlp",
        "erp_spatial_adapter",
        "panovggt_mlp",
    )

    def __init__(
        self,
        *args,
        module_learning_rates=None,
        trace_enable: bool = False,
        trace_sampler_shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.module_learning_rates = {
            name: float(lr)
            for name, lr in (module_learning_rates or {}).items()
            if lr is not None
        }
        self.trace_enable = bool(trace_enable)
        self.trace_sampler_shuffle = bool(trace_sampler_shuffle)

    def _get_train_sampler(self, train_dataset=None):
        if not self.trace_enable:
            return super()._get_train_sampler(train_dataset)

        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None

        episode_keys = getattr(train_dataset, "episode_keys", None)
        step_indices = getattr(train_dataset, "step_indices", None)
        end_step_indices = getattr(train_dataset, "end_step_indices", None)
        action_sequences = getattr(train_dataset, "action_sequences", None)
        if (
            episode_keys is None
            or step_indices is None
            or end_step_indices is None
            or action_sequences is None
        ):
            raise ValueError(
                "trace_enable=true requires train_dataset to be built with "
                "collect_trace_metadata=true"
            )

        per_process_batch_size = max(1, int(self._train_batch_size))
        world_size = max(1, int(self.args.world_size))
        gradient_accumulation_steps = max(1, int(self.args.gradient_accumulation_steps))
        window_size = per_process_batch_size * world_size * gradient_accumulation_steps

        sampler = TraceSampler(
            episode_keys=episode_keys,
            step_indices=step_indices,
            end_step_indices=end_step_indices,
            action_sequences=action_sequences,
            window_size=window_size,
            conflict_step_window=4,
            seed=self.args.seed,
            shuffle=self.trace_sampler_shuffle,
        )
        if RANK == 0:
            rank0_print(
                RANK,
                "TRACE sampler enabled: "
                f"samples={len(sampler)}, "
                f"episode_groups={sampler.num_episode_groups}, "
                f"window_size={window_size}, "
                f"per_process_batch_size={per_process_batch_size}, "
                f"world_size={world_size}, "
                f"gradient_accumulation_steps={gradient_accumulation_steps}",
            )
        return sampler

    def get_decay_parameter_names(self, model):
        decay_parameter_names = super().get_decay_parameter_names(model)
        return [
            name
            for name in decay_parameter_names
            if not name.endswith(self.RAW_ALPHA_NO_DECAY_SUFFIXES)
        ]

    @staticmethod
    def _name_has_module(name: str, module_name: str) -> bool:
        return name == module_name or name.startswith(f"{module_name}.") or f".{module_name}." in name

    def _module_lr_key_for_parameter(self, name: str):
        if self._name_has_module(name, "erp_position_mlp"):
            return "erp_position_mlp"
        if self._name_has_module(name, "erp_spatial_adapter"):
            return "erp_spatial_adapter"
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
                group = {
                    "params": [],
                    "weight_decay": weight_decay,
                    "lr": lr,
                }
                grouped_parameters[group_key] = group
            grouped_parameters[group_key]["params"].append(param)

        optimizer_grouped_parameters = list(grouped_parameters.values())
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


def copy_chat_template_files(source_dir: str, output_dir: str):
    for template_name in ("chat_template.json", "chat_template.jinja"):
        source_path = os.path.join(source_dir, template_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(output_dir, template_name))


def _config_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def apply_config_overrides(cfg, overrides):
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
        setattr(target, field_name, yaml.safe_load(raw_value))


def print_training_config(cfg) -> None:
    rank0_print(RANK, "===== Ablation config =====")
    rank0_print(RANK, f"torch_dtype: {_config_value(cfg.model.torch_dtype)}")
    rank0_print(RANK, f"attn_implementation: {_config_value(cfg.model.attn_implementation)}")
    rank0_print(RANK, "trainable_modules:")
    for name, enabled in (cfg.model.trainable_modules or {}).items():
        rank0_print(RANK, f"  {name}: {_config_value(enabled)}")
    rank0_print(RANK, f"erp_pos_enabled: {_config_value(cfg.model.erp_pos_enabled)}")
    rank0_print(RANK, f"erp_pos_alpha_value: {_config_value(cfg.model.erp_pos_alpha_value)}")
    rank0_print(RANK, f"erp_apply_to_current_only: {_config_value(cfg.model.erp_apply_to_current_only)}")
    rank0_print(RANK, f"erp_top_crop_degrees: {_config_value(cfg.model.erp_top_crop_degrees)}")
    rank0_print(RANK, f"erp_bottom_crop_degrees: {_config_value(cfg.model.erp_bottom_crop_degrees)}")
    rank0_print(RANK, f"erp_spatial_enabled: {_config_value(cfg.model.erp_spatial_enabled)}")
    rank0_print(RANK, f"erp_spatial_alpha_value: {_config_value(cfg.model.erp_spatial_alpha_value)}")
    rank0_print(RANK, f"panovggt_enabled: {_config_value(cfg.model.panovggt_enabled)}")
    rank0_print(RANK, f"panovggt_alpha_value: {_config_value(cfg.model.panovggt_alpha_value)}")
    rank0_print(RANK, f"panovggt_sampling_mode: {_config_value(cfg.model.panovggt_sampling_mode)}")
    rank0_print(RANK, f"panovggt_force_fp32: {_config_value(cfg.model.panovggt_force_fp32)}")
    rank0_print(RANK, f"data_shuffle: {_config_value(cfg.data.shuffle)}")
    rank0_print(RANK, f"trace_enable: {_config_value(cfg.data.trace_enable)}")
    rank0_print(RANK, f"per_device_train_batch_size: {cfg.training.per_device_train_batch_size}")
    rank0_print(RANK, f"gradient_accumulation_steps: {cfg.training.gradient_accumulation_steps}")
    rank0_print(RANK, f"learning_rate: {cfg.training.learning_rate}")
    rank0_print(RANK, f"language_model_lr: {_config_value(cfg.training.language_model_lr)}")
    rank0_print(RANK, f"visual_lr: {_config_value(cfg.training.visual_lr)}")
    rank0_print(RANK, f"visual_merger_lr: {_config_value(cfg.training.visual_merger_lr)}")
    rank0_print(RANK, f"erp_position_mlp_lr: {_config_value(cfg.training.erp_position_mlp_lr)}")
    rank0_print(RANK, f"erp_spatial_lr: {_config_value(cfg.training.erp_spatial_lr)}")
    rank0_print(RANK, f"panovggt_mlp_lr: {_config_value(cfg.training.panovggt_mlp_lr)}")
    rank0_print(RANK, f"bf16: {_config_value(cfg.training.bf16)}")
    rank0_print(RANK, f"fp16: {_config_value(cfg.training.fp16)}")
    rank0_print(RANK, "===========================")


def safe_save_model_for_hf_trainer(
    trainer: Trainer,
    output_dir: str,
    max_shard_size: str = "5GB",
):
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


def main():
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
    if RANK == 0:
        print_training_config(cfg)
    set_seed(cfg.training.seed)

    processor, tokenizer = load_processor_and_tokenizer(cfg)
    model = load_model(cfg)
    model_config = model.config
    vision_config = getattr(model_config, "vision_config", None)
    effective_panovggt_enabled = bool(getattr(model_config, "panovggt_enabled", cfg.model.panovggt_enabled))
    effective_erp_top_crop_degrees = float(
        getattr(model_config, "erp_top_crop_degrees", cfg.model.erp_top_crop_degrees)
    )
    effective_erp_bottom_crop_degrees = float(
        getattr(model_config, "erp_bottom_crop_degrees", cfg.model.erp_bottom_crop_degrees)
    )
    if RANK == 0:
        rank0_print(RANK, "===== Effective model config =====")
        rank0_print(RANK, f"panovggt_enabled: {_config_value(effective_panovggt_enabled)}")
        rank0_print(RANK, f"panovggt_alpha_value: {_config_value(getattr(model_config, 'panovggt_alpha_value', None))}")
        rank0_print(RANK, f"panovggt_sampling_mode: {_config_value(getattr(model_config, 'panovggt_sampling_mode', None))}")
        rank0_print(RANK, f"erp_top_crop_degrees: {_config_value(effective_erp_top_crop_degrees)}")
        rank0_print(RANK, f"erp_bottom_crop_degrees: {_config_value(effective_erp_bottom_crop_degrees)}")
        rank0_print(
            RANK,
            f"erp_spatial_enabled: {_config_value(getattr(vision_config, 'erp_spatial_enabled', None))}",
        )
        rank0_print(
            RANK,
            f"erp_spatial_alpha_value: {_config_value(getattr(vision_config, 'erp_spatial_alpha_value', None))}",
        )
        rank0_print(RANK, "==================================")
    train_image_root = cfg.data.train_image_root
    eval_image_root = cfg.data.eval_image_root or train_image_root

    train_dataset = SupervisedDataset(
        jsonl_path=cfg.data.train_jsonl,
        processor=processor,
        tokenizer=tokenizer,
        image_root=train_image_root,
        image_token=cfg.model.image_token,
        model_max_length=cfg.model.model_max_length,
        erp_top_crop_degrees=effective_erp_top_crop_degrees,
        erp_bottom_crop_degrees=effective_erp_bottom_crop_degrees,
        panovggt_enabled=effective_panovggt_enabled,
        max_samples=cfg.data.train_max_samples,
        shuffle=cfg.data.shuffle,
        prompt_format=cfg.data.prompt_format,
        collect_trace_metadata=cfg.data.trace_enable,
    )

    eval_dataset = None
    if cfg.data.eval_jsonl and cfg.run.do_eval:
        eval_dataset = SupervisedDataset(
            jsonl_path=cfg.data.eval_jsonl,
            processor=processor,
            tokenizer=tokenizer,
            image_root=eval_image_root,
            image_token=cfg.model.image_token,
            model_max_length=cfg.model.model_max_length,
            erp_top_crop_degrees=effective_erp_top_crop_degrees,
            erp_bottom_crop_degrees=effective_erp_bottom_crop_degrees,
            panovggt_enabled=effective_panovggt_enabled,
            max_samples=cfg.data.eval_max_samples,
            shuffle=True,
            prompt_format=cfg.data.prompt_format,
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
            cfg.training.eval_strategy if cfg.run.do_eval else "no"
        ),
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
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

    sync_model_special_tokens(model, tokenizer)
    set_model(cfg, model)

    if RANK == 0:
        print_model_parameters(model)

    init_wandb(cfg.wandb, training_args, RANK)

    trainer = PanoVLNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultiModalDataCollator(tokenizer),
        processing_class=tokenizer,
        trace_enable=cfg.data.trace_enable,
        trace_sampler_shuffle=cfg.data.shuffle,
        module_learning_rates={
            "language_model": cfg.training.language_model_lr,
            "visual": cfg.training.visual_lr,
            "visual_merger": cfg.training.visual_merger_lr,
            "erp_position_mlp": cfg.training.erp_position_mlp_lr,
            "erp_spatial_adapter": cfg.training.erp_spatial_lr,
            "panovggt_mlp": cfg.training.panovggt_mlp_lr,
        },
        compute_metrics=(
            build_action_accuracy(
                tokenizer,
                cfg.data.action_vocab,
                cfg.data.f1_action_weight,
            )
            if eval_dataset is not None else None
        ),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics
            if eval_dataset is not None else None
        ),
    )

    rank0_print(RANK, "calling trainer.train()")

    resume_enabled = bool(getattr(cfg.run, "resume_from_checkpoint", False))
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if resume_enabled and last_checkpoint:
        rank0_print(RANK, f"resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if resume_enabled and not last_checkpoint:
            rank0_print(RANK, "resume requested but no checkpoint found, start training")
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

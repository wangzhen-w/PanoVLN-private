import argparse
import os
import shutil

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from config.config import load_config
from data.collator import MultiModalDataCollator
from data.data import SupervisedDataset
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
        "panovggt_mlp.raw_alpha",
        "action_bearing_residual.raw_alpha",
    )

    def get_decay_parameter_names(self, model):
        decay_parameter_names = super().get_decay_parameter_names(model)
        return [
            name
            for name in decay_parameter_names
            if not name.endswith(self.RAW_ALPHA_NO_DECAY_SUFFIXES)
        ]


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


def print_training_config(cfg) -> None:
    rank0_print(RANK, "===== Ablation config =====")
    rank0_print(RANK, f"torch_dtype: {_config_value(cfg.model.torch_dtype)}")
    rank0_print(RANK, f"attn_implementation: {_config_value(cfg.model.attn_implementation)}")
    rank0_print(RANK, "trainable_modules:")
    for name, enabled in (cfg.model.trainable_modules or {}).items():
        rank0_print(RANK, f"  {name}: {_config_value(enabled)}")
    rank0_print(RANK, f"panovggt_enabled: {_config_value(cfg.model.panovggt_enabled)}")
    rank0_print(RANK, f"panovggt_alpha_init: {_config_value(cfg.model.panovggt_alpha_init)}")
    rank0_print(RANK, f"panovggt_alpha_max: {_config_value(cfg.model.panovggt_alpha_max)}")
    rank0_print(RANK, f"action_bearing_enabled: {_config_value(cfg.model.action_bearing_enabled)}")
    rank0_print(RANK, f"action_bearing_alpha_init: {_config_value(cfg.model.action_bearing_alpha_init)}")
    rank0_print(RANK, f"action_bearing_alpha_max: {_config_value(cfg.model.action_bearing_alpha_max)}")
    rank0_print(RANK, f"per_device_train_batch_size: {cfg.training.per_device_train_batch_size}")
    rank0_print(RANK, f"gradient_accumulation_steps: {cfg.training.gradient_accumulation_steps}")
    rank0_print(RANK, f"learning_rate: {cfg.training.learning_rate}")
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    if RANK == 0:
        print_training_config(cfg)
    set_seed(cfg.training.seed)

    processor, tokenizer = load_processor_and_tokenizer(cfg)
    train_image_root = cfg.data.train_image_root
    eval_image_root = cfg.data.eval_image_root or train_image_root

    train_dataset = SupervisedDataset(
        jsonl_path=cfg.data.train_jsonl,
        processor=processor,
        tokenizer=tokenizer,
        image_root=train_image_root,
        image_token=cfg.model.image_token,
        model_max_length=cfg.model.model_max_length,
        image_size=cfg.data.image_size,
        panovggt_enabled=cfg.model.panovggt_enabled,
        max_samples=cfg.data.train_max_samples,
        shuffle=cfg.data.shuffle,
        prompt_format=cfg.data.prompt_format,
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
            image_size=cfg.data.image_size,
            panovggt_enabled=cfg.model.panovggt_enabled,
            max_samples=cfg.data.eval_max_samples,
            shuffle=True,
            prompt_format=cfg.data.prompt_format,
        )

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
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

    model = load_model(cfg)
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
        wandb = load_wandb_module(required=False)
        if wandb is not None and hasattr(wandb, "finish"):
            wandb.finish()


if __name__ == "__main__":
    main()

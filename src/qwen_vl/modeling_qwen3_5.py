import json
import math
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
)
from transformers.utils import can_return_tuple


PANOVGGT_AGGREGATOR_LAYER = -1
PANOVGGT_AGGREGATOR_CONTEXT_DIM = 2048
PANOVGGT_POINT_HIDDEN_DIM = 1024
PANOVGGT_FEATURE_SOURCES = {"aggregator", "point_hidden"}
PANOVGGT_INJECTION_STAGES = {"post_merger", "pre_merger"}
PANOVGGT_MLP_HIDDEN_SIZE = 4096
ACTION_BEARING_HIDDEN_SIZE = 2048
ACTION_BEARING_TURN_ANGLE_DEG = 15.0
ACTION_BEARING_MAX_STEPS = 4
ACTION_BEARING_SIGMA_STEPS = 0.6
ACTION_BEARING_LEFT_TOKEN_ID = 2282
ACTION_BEARING_RIGHT_TOKEN_ID = 1246
ACTION_BEARING_FRONT_TOKEN_ID = 6735
ACTION_BEARING_ONE_TOKEN_ID = 799
ACTION_BEARING_TWO_TOKEN_ID = 1330
ACTION_BEARING_THREE_TOKEN_ID = 2250
ACTION_BEARING_FOUR_TOKEN_ID = 2943
ACTION_BEARING_BIN_TOKEN_IDS = (
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_FOUR_TOKEN_ID),
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_THREE_TOKEN_ID),
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_TWO_TOKEN_ID),
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_ONE_TOKEN_ID),
    (ACTION_BEARING_FRONT_TOKEN_ID,),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_ONE_TOKEN_ID),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_TWO_TOKEN_ID),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_THREE_TOKEN_ID),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_FOUR_TOKEN_ID),
)
PAQR_DEFAULT_ACTION_TOKEN_IDS = (2282, 13048, 1246)  # left, forward, right
PAQR_DEFAULT_STOP_TOKEN_ID = 9215
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
VENDORED_PANOVGGT_DIR = SRC_ROOT / "panovggt"
VENDORED_PANOVGGT_CONFIG_PATH = VENDORED_PANOVGGT_DIR / "training" / "config" / "default.yaml"


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _inverse_tanh(value: float) -> float:
    value = min(max(float(value), -1.0 + 1e-6), 1.0 - 1e-6)
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


def _bounded_raw_alpha(alpha_init: float, alpha_max: float) -> torch.Tensor:
    alpha_max = float(alpha_max)
    alpha_init = float(alpha_init)
    if alpha_max <= 0.0:
        raise ValueError(f"alpha_max must be positive, got {alpha_max}")
    alpha_init = min(max(alpha_init, 1e-6), alpha_max * (1.0 - 1e-6))
    return torch.tensor(_inverse_sigmoid(alpha_init / alpha_max), dtype=torch.float32)


def ensure_action_bearing_config(config) -> None:
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(
        text_config,
        "hidden_size",
        getattr(vision_config, "out_hidden_size", 3584),
    )
    output_dim = int(getattr(vision_config, "out_hidden_size", text_hidden_size))
    defaults = {
        "action_bearing_enabled": False,
        "action_bearing_alpha_init": 0.02,
        "action_bearing_alpha_max": 0.1,
        "action_bearing_output_dim": output_dim,
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


class ActionBearingResidual(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_action_bearing_config(config)

        self.enabled = bool(getattr(config, "action_bearing_enabled", False))
        self.output_dim = int(
            getattr(config, "action_bearing_output_dim", config.text_config.hidden_size)
        )
        self.hidden_dim = ACTION_BEARING_HIDDEN_SIZE
        self.alpha_init = float(getattr(config, "action_bearing_alpha_init", 0.02))
        self.alpha_max = float(getattr(config, "action_bearing_alpha_max", 0.1))
        self.turn_angle_deg = ACTION_BEARING_TURN_ANGLE_DEG
        self.max_steps = ACTION_BEARING_MAX_STEPS
        self.sigma_steps = ACTION_BEARING_SIGMA_STEPS
        self.spatial_merge_size = int(getattr(config.vision_config, "spatial_merge_size", 2))

        if self.turn_angle_deg <= 0.0:
            raise ValueError(
                f"action_bearing_turn_angle_deg must be positive, got {self.turn_angle_deg}"
            )
        if self.max_steps < 1:
            raise ValueError(
                f"action_bearing_max_steps must be >= 1, got {self.max_steps}"
            )
        if self.sigma_steps <= 0.0:
            raise ValueError(
                f"action_bearing_sigma_steps must be positive, got {self.sigma_steps}"
            )

        self.num_bins = 2 * self.max_steps + 1
        if len(ACTION_BEARING_BIN_TOKEN_IDS) != self.num_bins:
            raise AssertionError(
                "ACTION_BEARING_BIN_TOKEN_IDS must match the action-bearing bin count: "
                f"token_id_rows={len(ACTION_BEARING_BIN_TOKEN_IDS)}, num_bins={self.num_bins}"
            )
        self.bin_embeddings = nn.Parameter(torch.empty(self.num_bins, self.output_dim))
        self.input_norm = nn.RMSNorm(self.output_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.output_norm = nn.RMSNorm(self.output_dim, eps=1e-6)
        self.raw_alpha = nn.Parameter(_bounded_raw_alpha(self.alpha_init, self.alpha_max))
        self.reset_parameters(float(getattr(config.vision_config, "initializer_range", 0.02)))

    @property
    def alpha(self) -> torch.Tensor:
        return float(self.alpha_max) * torch.sigmoid(self.raw_alpha)

    @property
    def turn_angle_radians(self) -> float:
        return math.radians(self.turn_angle_deg)

    def reset_parameters(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.bin_embeddings, mean=0.0, std=float(init_std))
        self.input_norm.reset_parameters()
        self.output_norm.reset_parameters()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=float(init_std))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.raw_alpha.copy_(_bounded_raw_alpha(self.alpha_init, self.alpha_max))

    def initialize_bin_embeddings_from_text(
        self,
        input_embeddings: nn.Embedding | None,
    ) -> bool:
        if input_embeddings is None or not hasattr(input_embeddings, "weight"):
            return False
        weight = input_embeddings.weight
        if int(weight.shape[1]) != self.output_dim:
            return False

        phrase_embeddings = []
        vocab_size = int(weight.shape[0])
        for phrase_token_ids in ACTION_BEARING_BIN_TOKEN_IDS:
            token_ids = [
                self._valid_token_id(token_id, vocab_size)
                for token_id in phrase_token_ids
            ]
            if any(token_id is None for token_id in token_ids):
                return False
            ids = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
            phrase_embeddings.append(weight.index_select(0, ids).mean(dim=0))

        with torch.no_grad():
            init_values = torch.stack(phrase_embeddings, dim=0).to(
                device=self.bin_embeddings.device,
                dtype=self.bin_embeddings.dtype,
            )
            self.bin_embeddings.copy_(init_values)
        return True

    @staticmethod
    def _valid_token_id(token_id, vocab_size: int) -> int | None:
        if token_id is None:
            return None
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            return None
        return token_id

    def compute_soft_action_scores(self, yaw: torch.Tensor) -> torch.Tensor:
        steps = (yaw.to(dtype=torch.float32) / float(self.turn_angle_radians)).clamp(
            min=-float(self.max_steps),
            max=float(self.max_steps),
        )
        centers = torch.arange(
            -self.max_steps,
            self.max_steps + 1,
            device=steps.device,
            dtype=torch.float32,
        )
        distances = steps.unsqueeze(-1) - centers
        scores = torch.exp(-0.5 * (distances / float(self.sigma_steps)).pow(2))
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _build_grid_yaw(target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
        del target_h
        xs = torch.arange(target_w, device=device, dtype=torch.float32) + 0.5
        yaw = (xs / float(target_w) - 0.5) * (2.0 * math.pi)
        return yaw.unsqueeze(0)

    def _target_grid_hw(
        self,
        grid_thw: list[int],
        target_len: int,
    ) -> tuple[int, int]:
        num_frames, grid_h, grid_w = [int(value) for value in grid_thw]
        if num_frames != 1:
            raise AssertionError(
                "Action-Bearing residual expects a single current panorama per Qwen image, "
                f"got image_grid_thw={grid_thw}"
            )
        target_h = max(1, grid_h // self.spatial_merge_size)
        target_w = max(1, grid_w // self.spatial_merge_size)
        expected_len = target_h * target_w
        if expected_len != int(target_len):
            raise AssertionError(
                "Cannot map Action-Bearing scores to the Qwen 2D visual-token grid: "
                f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}, "
                f"expected_len={expected_len}, qwen_len={int(target_len)}"
            )
        return target_h, target_w

    def forward(
        self,
        image_tokens: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled or image_tokens.numel() == 0:
            return image_tokens

        target_h, target_w = self._target_grid_hw(
            grid_thw.detach().to(device="cpu", dtype=torch.long).tolist(),
            target_len=int(image_tokens.shape[0]),
        )
        param = self.bin_embeddings
        yaw = self._build_grid_yaw(
            target_h,
            target_w,
            device=param.device,
        ).expand(target_h, target_w)
        scores = self.compute_soft_action_scores(yaw.reshape(-1)).to(dtype=param.dtype)
        action_prior = scores @ self.bin_embeddings
        delta = self.output_norm(self.mlp(self.input_norm(action_prior)))
        delta = self.alpha.to(dtype=delta.dtype) * delta
        return image_tokens + delta.to(
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )


def ensure_paqr_config(config) -> None:
    defaults = {
        "paqr_enabled": False,
        "paqr_action_token_ids": list(PAQR_DEFAULT_ACTION_TOKEN_IDS),
        "paqr_stop_token_id": PAQR_DEFAULT_STOP_TOKEN_ID,
        "paqr_temperature": 0.1,
        "paqr_prior_init": 0.02,
        "paqr_prior_max": 0.25,
        "paqr_logit_scale_max": 0.5,
        "paqr_first_action_only": True,
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


class PanoramicActionQueryReadout(nn.Module):
    """Read directional evidence from current-panorama tokens with LM-head queries."""

    def __init__(self, config) -> None:
        super().__init__()
        ensure_paqr_config(config)

        action_token_ids = tuple(
            int(token_id) for token_id in getattr(config, "paqr_action_token_ids")
        )
        if len(action_token_ids) != 3 or len(set(action_token_ids)) != 3:
            raise ValueError(
                "paqr_action_token_ids must contain three distinct ids in "
                f"[left, forward, right] order, got {action_token_ids}"
            )

        self.enabled = bool(getattr(config, "paqr_enabled", False))
        self.temperature = float(getattr(config, "paqr_temperature", 0.1))
        self.prior_init = float(getattr(config, "paqr_prior_init", 0.02))
        self.prior_max = float(getattr(config, "paqr_prior_max", 0.25))
        self.logit_scale_max = float(
            getattr(config, "paqr_logit_scale_max", 0.5)
        )
        self.first_action_only = bool(
            getattr(config, "paqr_first_action_only", True)
        )
        if self.temperature <= 0.0:
            raise ValueError(
                f"paqr_temperature must be positive, got {self.temperature}"
            )
        if self.prior_max <= 0.0:
            raise ValueError(f"paqr_prior_max must be positive, got {self.prior_max}")
        if not 0.0 <= self.prior_init < self.prior_max:
            raise ValueError(
                "paqr_prior_init must satisfy 0 <= init < max, got "
                f"init={self.prior_init}, max={self.prior_max}"
            )
        if self.logit_scale_max <= 0.0:
            raise ValueError(
                "paqr_logit_scale_max must be positive, got "
                f"{self.logit_scale_max}"
            )
        if not self.first_action_only:
            raise ValueError(
                "This PAQR implementation is deliberately restricted to the first "
                "action token; set paqr_first_action_only=true"
            )

        # Keep vocabulary ids as immutable Python metadata. Tiny non-persistent
        # integer buffers can be coalesced incorrectly by some ZeRO setups.
        self.action_token_id_values = action_token_ids
        self.raw_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.raw_turn_cos = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.raw_turn_sin = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.raw_forward_cos = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    @property
    def gate(self) -> torch.Tensor:
        return float(self.logit_scale_max) * torch.tanh(self.raw_gate.float())

    @property
    def action_token_ids(self) -> torch.Tensor:
        return torch.tensor(self.action_token_id_values, dtype=torch.long)

    def reset_parameters(self) -> None:
        # The physical first-harmonic initialization is a fixed center; the
        # trainable values are zero-centered residuals. Besides making the
        # prior easy to interpret, this preserves small updates when the rest
        # of the model is loaded in bf16.
        with torch.no_grad():
            self.raw_gate.zero_()
            self.raw_turn_cos.zero_()
            self.raw_turn_sin.zero_()
            self.raw_forward_cos.zero_()

    def circular_prior(self, yaw: torch.Tensor) -> torch.Tensor:
        """Return bounded first-harmonic priors in [left, forward, right] order."""
        yaw = yaw.to(device=self.raw_gate.device, dtype=torch.float32)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        base_amplitude = _inverse_tanh(self.prior_init / self.prior_max)
        turn_angle = math.radians(ACTION_BEARING_TURN_ANGLE_DEG)
        turn_cos = (
            base_amplitude * math.cos(turn_angle) + self.raw_turn_cos.float()
        )
        turn_sin = (
            base_amplitude * math.sin(turn_angle) + self.raw_turn_sin.float()
        )
        forward_cos = base_amplitude + self.raw_forward_cos.float()
        left_raw = turn_cos * cos_yaw - turn_sin * sin_yaw
        forward_raw = forward_cos * cos_yaw
        right_raw = turn_cos * cos_yaw + turn_sin * sin_yaw
        return float(self.prior_max) * torch.tanh(
            torch.stack((left_raw, forward_raw, right_raw), dim=0)
        )

    def compute_evidence(
        self,
        panorama_hidden: torch.Tensor,
        yaw: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if panorama_hidden.ndim != 2:
            raise ValueError(
                "panorama_hidden must have shape [num_tokens, hidden_size], got "
                f"{tuple(panorama_hidden.shape)}"
            )
        if int(panorama_hidden.shape[0]) != int(yaw.numel()):
            raise ValueError(
                "PAQR yaw/token length mismatch: "
                f"tokens={panorama_hidden.shape[0]}, yaw={yaw.numel()}"
            )
        if int(panorama_hidden.shape[1]) != int(lm_head_weight.shape[1]):
            raise ValueError(
                "PAQR hidden size does not match lm_head: "
                f"hidden={panorama_hidden.shape[1]}, lm_head={lm_head_weight.shape[1]}"
            )

        action_token_ids = self.action_token_ids.to(device=lm_head_weight.device)
        if int(action_token_ids.max().item()) >= int(lm_head_weight.shape[0]):
            raise ValueError(
                "PAQR action token id exceeds lm_head vocabulary size: "
                f"ids={action_token_ids.tolist()}, vocab={lm_head_weight.shape[0]}"
            )

        # These are the exact live LM-head rows. Detaching only this auxiliary
        # query path prevents PAQR from learning by distorting the tied token
        # embeddings; the normal language-model loss still trains lm_head.
        action_queries = lm_head_weight.index_select(0, action_token_ids).detach()
        action_queries = F.normalize(action_queries.float(), dim=-1, eps=1e-6)
        panorama_tokens = F.normalize(
            panorama_hidden.float(),
            dim=-1,
            eps=1e-6,
        )
        content_scores = action_queries @ panorama_tokens.transpose(0, 1)
        prior = self.circular_prior(yaw).to(device=content_scores.device)
        attention = torch.softmax(
            content_scores / float(self.temperature) + prior,
            dim=-1,
        )
        evidence = (attention * content_scores).sum(dim=-1)
        return evidence, attention, content_scores, prior

    def forward(
        self,
        panorama_hidden: torch.Tensor,
        yaw: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled or panorama_hidden.numel() == 0:
            return panorama_hidden.new_zeros((3,), dtype=torch.float32)

        evidence, _, _, _ = self.compute_evidence(
            panorama_hidden=panorama_hidden,
            yaw=yaw,
            lm_head_weight=lm_head_weight,
        )
        # Remove evidence common to all three bearings: PAQR should express
        # which movement direction the panorama supports, not add a shared
        # "move" bias. The stop logit itself is never written; as with any
        # directional-logit change, its normalized probability can still move.
        centered_evidence = evidence - evidence.mean()
        return self.gate.to(device=evidence.device) * centered_evidence

    def diagnostics(self) -> dict[str, float]:
        with torch.no_grad():
            return {
                "paqr/gate": float(self.gate.detach().float().cpu().item()),
                "paqr/prior_turn_cos_raw": float(
                    self.raw_turn_cos.detach().float().cpu().item()
                ),
                "paqr/prior_turn_sin_raw": float(
                    self.raw_turn_sin.detach().float().cpu().item()
                ),
                "paqr/prior_forward_cos_raw": float(
                    self.raw_forward_cos.detach().float().cpu().item()
                ),
            }


def ensure_panovggt_config(config) -> None:
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", getattr(vision_config, "out_hidden_size", 3584))
    defaults = {
        "panovggt_enabled": False,
        "panovggt_checkpoint_path": "/workspace/code/a_property/model/PanoVGGT/model.pt",
        "panovggt_alpha_value": 0.1,
        "panovggt_feature_source": "aggregator",
        "panovggt_injection_stage": "post_merger",
        "panovggt_sampling_mode": "grouping",
        "panovggt_force_fp32": False,
        "panovggt_output_dim": int(getattr(vision_config, "out_hidden_size", text_hidden_size)),
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


def _ensure_vendored_panovggt_available() -> None:
    if not VENDORED_PANOVGGT_DIR.exists():
        raise FileNotFoundError(
            "PanoVGGT is enabled but the vendored source directory is missing: "
            f"{VENDORED_PANOVGGT_DIR}"
        )
    if not VENDORED_PANOVGGT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "PanoVGGT is enabled but the vendored config is missing: "
            f"{VENDORED_PANOVGGT_CONFIG_PATH}"
        )

    src_root = str(SRC_ROOT)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def normalize_panovggt_feature_source(feature_source: str) -> str:
    feature_source = str(feature_source).lower()
    if feature_source not in PANOVGGT_FEATURE_SOURCES:
        raise ValueError(
            "panovggt_feature_source must be 'aggregator' or 'point_hidden', "
            f"got {feature_source!r}"
        )
    return feature_source


def normalize_panovggt_injection_stage(injection_stage: str) -> str:
    injection_stage = str(injection_stage).lower()
    if injection_stage not in PANOVGGT_INJECTION_STAGES:
        raise ValueError(
            "panovggt_injection_stage must be 'post_merger' or 'pre_merger', "
            f"got {injection_stage!r}"
        )
    return injection_stage


def build_panovggt_model_from_vendored_config(feature_source: str = "aggregator"):
    _ensure_vendored_panovggt_available()
    feature_source = normalize_panovggt_feature_source(feature_source)
    enable_point = feature_source == "point_hidden"

    try:
        from omegaconf import OmegaConf
        from panovggt.models import aggregator as panovggt_aggregator
        from panovggt.models.panovggt_model import PanoVGGTModel
    except Exception as exc:
        raise ImportError(
            "PanoVGGT is enabled but could not be imported from "
            f"{VENDORED_PANOVGGT_DIR}"
        ) from exc

    cfg = OmegaConf.load(str(VENDORED_PANOVGGT_CONFIG_PATH))
    OmegaConf.resolve(cfg)
    mc = cfg.model
    original_dinov2_loader = getattr(panovggt_aggregator.Aggregator, "_try_load_dinov2", None)
    if original_dinov2_loader is not None:
        panovggt_aggregator.Aggregator._try_load_dinov2 = lambda *args, **kwargs: None
    try:
        model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=False,
            enable_depth=False,
            enable_point=enable_point,
            enable_global_points=False,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        )
    finally:
        if original_dinov2_loader is not None:
            panovggt_aggregator.Aggregator._try_load_dinov2 = original_dinov2_loader
    freeze_panovggt_model(model)
    return model


def freeze_panovggt_model(model: nn.Module | None) -> None:
    if model is None:
        return
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_panovggt_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected PanoVGGT checkpoint format: {type(ckpt)!r}")
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in ckpt.items()
    }
    load_result = model.load_state_dict(state_dict, strict=False)
    required_prefixes = ["aggregator."]
    if bool(getattr(model, "enable_point", False)):
        required_prefixes.extend(("point_decoder.", "pos_adapters.point."))
    missing_required_keys = [
        key
        for key in getattr(load_result, "missing_keys", [])
        if any(key.startswith(prefix) for prefix in required_prefixes)
    ]
    if missing_required_keys:
        raise RuntimeError(
            "PanoVGGT checkpoint is missing required weights, "
            f"first missing key: {missing_required_keys[0]}"
        )
    freeze_panovggt_model(model)

class PanoVGGTGeometryMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_panovggt_config(config)

        self.enabled = bool(getattr(config, "panovggt_enabled", False))
        self.layer = PANOVGGT_AGGREGATOR_LAYER
        self.feature_source = normalize_panovggt_feature_source(
            getattr(config, "panovggt_feature_source", "aggregator")
        )
        self.injection_stage = normalize_panovggt_injection_stage(
            getattr(config, "panovggt_injection_stage", "post_merger")
        )
        self.context_dim = (
            PANOVGGT_POINT_HIDDEN_DIM
            if self.feature_source == "point_hidden"
            else PANOVGGT_AGGREGATOR_CONTEXT_DIM
        )
        if self.injection_stage == "pre_merger":
            self.output_dim = int(getattr(config.vision_config, "hidden_size"))
        else:
            self.output_dim = int(getattr(config, "panovggt_output_dim", config.text_config.hidden_size))
        self.hidden_dim = PANOVGGT_MLP_HIDDEN_SIZE
        self.sampling_mode = str(getattr(config, "panovggt_sampling_mode", "grouping")).lower()
        if self.sampling_mode not in {"singlepoint", "grouping"}:
            raise ValueError(
                "panovggt_sampling_mode must be 'singlepoint' or 'grouping', "
                f"got {self.sampling_mode!r}"
            )
        self.alpha_init = float(getattr(config, "panovggt_alpha_value", 0.1))
        self.register_buffer(
            "alpha_value",
            torch.tensor(self.alpha_init, dtype=torch.float32),
        )
        self.spatial_merge_size = int(getattr(config.vision_config, "spatial_merge_size", 2))
        if self.spatial_merge_size <= 0:
            raise ValueError(f"Qwen spatial_merge_size must be positive, got {self.spatial_merge_size}")
        self.spatial_merge_unit = self.spatial_merge_size**2
        if self.sampling_mode == "grouping":
            self.mlp_input_dim = self.context_dim * self.spatial_merge_unit
        else:
            self.mlp_input_dim = self.context_dim

        self.input_norm = nn.RMSNorm(self.context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.output_norm = nn.RMSNorm(self.output_dim, eps=1e-6)
        self.reset_parameters()

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_value

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        self.output_norm.reset_parameters()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.alpha_value.fill_(self.alpha_init)

    def _infer_patch_grid(
        self,
        *,
        patch_count: int,
        image_height: int,
        image_width: int,
        patch_size: int,
    ) -> tuple[int, int]:
        if patch_size > 0:
            patch_h = image_height // patch_size
            patch_w = image_width // patch_size
            if patch_h > 0 and patch_w > 0 and patch_h * patch_w == patch_count:
                return patch_h, patch_w

        aspect = max(float(image_width) / max(float(image_height), 1.0), 1e-6)
        best_pair = None
        best_error = float("inf")
        for patch_h in range(1, int(math.sqrt(patch_count)) + 2):
            if patch_count % patch_h != 0:
                continue
            patch_w = patch_count // patch_h
            error = abs((float(patch_w) / float(patch_h)) - aspect)
            if error < best_error:
                best_error = error
                best_pair = (patch_h, patch_w)
        if best_pair is None:
            raise AssertionError(f"Cannot infer PanoVGGT patch grid for {patch_count} tokens")
        return best_pair

    def _sample_geometry_grid(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
    ) -> torch.Tensor:
        if target_h <= 0 or target_w <= 0:
            return source_grid.new_zeros((0, source_grid.shape[1]))

        if geometry is None:
            vertical_fov = math.pi
            center_latitude = 0.0
        else:
            vertical_fov = float(geometry[0].detach().cpu())
            center_latitude = float(geometry[1].detach().cpu())

        ys = torch.arange(target_h, device=source_grid.device, dtype=torch.float32) + 0.5
        xs = torch.arange(target_w, device=source_grid.device, dtype=torch.float32) + 0.5
        lat = (ys[:, None] / float(target_h) - 0.5) * vertical_fov + center_latitude
        lon = (xs[None, :] / float(target_w) - 0.5) * (2.0 * math.pi)
        lat = lat.expand(target_h, target_w)
        lon = lon.expand(target_h, target_w)
        grid = torch.stack(
            [
                (lon / math.pi).clamp(-1.0 + 1e-6, 1.0 - 1e-6),
                (lat / (0.5 * math.pi)).clamp(-1.0 + 1e-6, 1.0 - 1e-6),
            ],
            dim=-1,
        ).unsqueeze(0)
        sampled = F.grid_sample(
            source_grid.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled[0].permute(1, 2, 0).reshape(target_h * target_w, source_grid.shape[1])

    def _sample_token_geometry_singlepoint(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sampled = self._sample_geometry_grid(
            source_grid,
            target_h=target_h,
            target_w=target_w,
            geometry=geometry,
        )
        return self.input_norm(sampled.to(dtype=dtype))

    def _sample_token_geometry_grouping(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        group_h = target_h * self.spatial_merge_size
        group_w = target_w * self.spatial_merge_size
        sampled = self._sample_geometry_grid(
            source_grid,
            target_h=group_h,
            target_w=group_w,
            geometry=geometry,
        )
        sampled = self.input_norm(sampled.to(dtype=dtype))
        sampled = sampled.reshape(
            target_h,
            self.spatial_merge_size,
            target_w,
            self.spatial_merge_size,
            self.context_dim,
        )
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()
        return sampled.reshape(target_h * target_w, self.mlp_input_dim)

    def _sample_token_geometry(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.sampling_mode == "grouping":
            return self._sample_token_geometry_grouping(
                source_grid,
                target_h=target_h,
                target_w=target_w,
                geometry=geometry,
                dtype=dtype,
            )
        return self._sample_token_geometry_singlepoint(
            source_grid,
            target_h=target_h,
            target_w=target_w,
            geometry=geometry,
            dtype=dtype,
        )

    def _to_qwen_pre_merger_order(
        self,
        tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
            raise AssertionError(
                "PanoVGGT pre-merger target grid is not divisible by Qwen spatial_merge_size: "
                f"grid=({grid_h}, {grid_w}), spatial_merge_size={self.spatial_merge_size}"
            )
        if int(tokens.shape[0]) != int(grid_h * grid_w):
            raise AssertionError(
                "PanoVGGT pre-merger token length mismatch before Qwen order conversion: "
                f"tokens={int(tokens.shape[0])}, grid=({grid_h}, {grid_w})"
            )
        tokens = tokens.reshape(
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            tokens.shape[-1],
        )
        tokens = tokens.permute(0, 2, 1, 3, 4).contiguous()
        return tokens.reshape(grid_h * grid_w, -1)

    def _point_decoder_xpos(
        self,
        *,
        encoder: nn.Module,
        pos_2d: torch.Tensor | None,
        batch_frames: int,
        token_count: int,
        patch_start_idx: int,
        patch_h: int,
        patch_w: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if getattr(encoder.aggregator, "rope", None) is None:
            return None
        if pos_2d is not None:
            if int(pos_2d.shape[0]) != batch_frames or int(pos_2d.shape[1]) != token_count:
                raise AssertionError(
                    "PanoVGGT point decoder xpos shape mismatch: "
                    f"xpos={tuple(pos_2d.shape)}, expected=({batch_frames}, {token_count}, 2)"
                )
            return pos_2d
        if not hasattr(encoder.aggregator, "position_getter"):
            raise AssertionError("PanoVGGT point decoder requires aggregator.position_getter for RoPE xpos")
        pos_2d = encoder.aggregator.position_getter(batch_frames, patch_h, patch_w, device)
        pos_2d = pos_2d + 1
        pos_special = torch.zeros(
            batch_frames,
            patch_start_idx,
            2,
            device=device,
            dtype=pos_2d.dtype,
        )
        return torch.cat([pos_special, pos_2d], dim=1)

    def _decode_point_hidden_tokens(
        self,
        *,
        encoder: nn.Module,
        tokens: torch.Tensor,
        patch_start_idx: int,
        image_h: int,
        image_w: int,
        pos_2d: torch.Tensor | None,
    ) -> torch.Tensor:
        if not hasattr(encoder, "point_decoder") or not hasattr(encoder, "_get_branch_pos_embed"):
            raise AssertionError(
                "panovggt_feature_source='point_hidden' requires PanoVGGT to be built with enable_point=True"
            )
        batch_size, num_frames, token_count, token_dim = [int(value) for value in tokens.shape]
        if token_dim != PANOVGGT_AGGREGATOR_CONTEXT_DIM:
            raise AssertionError(
                "PanoVGGT point decoder expects aggregator tokens with "
                f"{PANOVGGT_AGGREGATOR_CONTEXT_DIM} channels, got {token_dim}"
            )
        patch_size = int(getattr(encoder, "patch_size", 0))
        patch_count = token_count - int(patch_start_idx)
        patch_h, patch_w = self._infer_patch_grid(
            patch_count=patch_count,
            image_height=int(image_h),
            image_width=int(image_w),
            patch_size=patch_size,
        )
        batch_frames = batch_size * num_frames
        decoder_tokens = tokens.reshape(batch_frames, token_count, token_dim)
        decoder_xpos = self._point_decoder_xpos(
            encoder=encoder,
            pos_2d=pos_2d,
            batch_frames=batch_frames,
            token_count=token_count,
            patch_start_idx=patch_start_idx,
            patch_h=patch_h,
            patch_w=patch_w,
            device=decoder_tokens.device,
        )
        pos_embed = encoder._get_branch_pos_embed(
            patch_h,
            patch_w,
            patch_start_idx,
            decoder_tokens.device,
            decoder_tokens.dtype,
            "point",
            batch_frames,
        )
        point_hidden = encoder.point_decoder(decoder_tokens, pos_embed=pos_embed, xpos=decoder_xpos)
        if point_hidden.ndim != 3:
            raise AssertionError(
                f"Expected PanoVGGT point hidden [B*S, P, C], got {tuple(point_hidden.shape)}"
            )
        if int(point_hidden.shape[0]) != batch_frames or int(point_hidden.shape[1]) != token_count:
            raise AssertionError(
                "PanoVGGT point hidden shape mismatch: "
                f"point_hidden={tuple(point_hidden.shape)}, expected=({batch_frames}, {token_count}, C)"
            )
        return point_hidden.reshape(batch_size, num_frames, token_count, int(point_hidden.shape[-1]))

    def forward(
        self,
        panovggt_pixel_values: torch.Tensor,
        panovggt_model: nn.Module,
        target_grid_thw: torch.Tensor,
        target_lengths: list[int],
        image_erp_geometry: torch.Tensor | None,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if (
            not self.enabled
            or panovggt_model is None
            or panovggt_pixel_values is None
            or panovggt_pixel_values.numel() == 0
        ):
            return []

        if panovggt_pixel_values.ndim == 4:
            panovggt_pixel_values = panovggt_pixel_values.unsqueeze(1)
        if panovggt_pixel_values.ndim != 5:
            raise AssertionError(
                "Expected panovggt_pixel_values shape [B, S, 3, H, W] "
                f"or [B, 3, H, W], got {tuple(panovggt_pixel_values.shape)}"
            )

        encoder = panovggt_model
        encoder_param = next(encoder.parameters())
        param = next(self.mlp.parameters())
        images = panovggt_pixel_values.to(device=encoder_param.device, dtype=encoder_param.dtype)

        with torch.no_grad():
            aggregated = encoder.aggregator(images)
            pos_2d = None
            if isinstance(aggregated, (list, tuple)):
                token_list = aggregated[0]
                patch_start_idx = int(aggregated[1])
                pos_2d = aggregated[2] if len(aggregated) > 2 else None
                tokens = token_list[self.layer] if isinstance(token_list, list) else token_list
            else:
                tokens = aggregated
                patch_start_idx = 0

            if tokens.ndim != 4:
                raise AssertionError(f"Expected PanoVGGT tokens [B, S, P, C], got {tuple(tokens.shape)}")
            _, _, _, image_h, image_w = images.shape
            if self.feature_source == "point_hidden":
                tokens = self._decode_point_hidden_tokens(
                    encoder=encoder,
                    tokens=tokens,
                    patch_start_idx=patch_start_idx,
                    image_h=int(image_h),
                    image_w=int(image_w),
                    pos_2d=pos_2d,
                )
        tokens = tokens[:, -1, patch_start_idx:, :]
        if tokens.shape[-1] != self.context_dim:
            raise AssertionError(
                f"PanoVGGT context dim mismatch: expected {self.context_dim}, got {tokens.shape[-1]}"
            )

        _, _, _, image_h, image_w = images.shape
        patch_size = int(getattr(encoder, "patch_size", 0))
        patch_h, patch_w = self._infer_patch_grid(
            patch_count=int(tokens.shape[1]),
            image_height=int(image_h),
            image_width=int(image_w),
            patch_size=patch_size,
        )
        source_grid = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

        deltas = []
        geometry = image_erp_geometry
        if geometry is not None:
            geometry = geometry.to(device=param.device, dtype=torch.float32)
        if int(source_grid.shape[0]) != len(target_lengths) or len(target_lengths) != int(target_grid_thw.shape[0]):
            raise AssertionError(
                "PanoVGGT batch size mismatch: "
                f"source={int(source_grid.shape[0])}, target_lengths={len(target_lengths)}, "
                f"target_grid_thw={int(target_grid_thw.shape[0])}"
            )
        if geometry is not None and int(geometry.shape[0]) != len(target_lengths):
            raise AssertionError(
                "PanoVGGT geometry batch size mismatch: "
                f"geometry={int(geometry.shape[0])}, target_lengths={len(target_lengths)}"
            )
        for sample_index, (grid_thw, target_len) in enumerate(zip(target_grid_thw.tolist(), target_lengths)):
            num_frames, grid_h, grid_w = [int(value) for value in grid_thw]
            if num_frames != 1:
                raise AssertionError(
                    "PanoVGGT geometry fusion expects a single current panorama per Qwen image, "
                    f"got image_grid_thw={grid_thw}"
                )
            if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
                raise AssertionError(
                    "PanoVGGT target grid is not divisible by Qwen spatial_merge_size: "
                    f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}"
                )
            if self.injection_stage == "pre_merger":
                target_h = grid_h
                target_w = grid_w
            else:
                target_h = max(1, grid_h // self.spatial_merge_size)
                target_w = max(1, grid_w // self.spatial_merge_size)
            expected_len = target_h * target_w
            if expected_len != int(target_len):
                raise AssertionError(
                    "Cannot map PanoVGGT geometry to the Qwen 2D visual-token grid: "
                    f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}, "
                    f"expected_len={expected_len}, qwen_len={int(target_len)}"
                )

            sample_geometry = None if geometry is None else geometry[sample_index]
            geo = self._sample_token_geometry(
                source_grid[sample_index:sample_index + 1],
                target_h=target_h,
                target_w=target_w,
                geometry=sample_geometry,
                dtype=param.dtype,
            )
            if geo.shape[0] != int(target_len):
                raise AssertionError(
                    "PanoVGGT sampled geometry length mismatch: "
                    f"sampled_len={geo.shape[0]}, qwen_len={int(target_len)}"
                )
            projected = self.output_norm(self.mlp(geo))
            projected = self.alpha.to(dtype=projected.dtype) * projected
            if self.injection_stage == "pre_merger":
                projected = self._to_qwen_pre_merger_order(projected, grid_h=grid_h, grid_w=grid_w)
            deltas.append(projected.to(device=output_device, dtype=output_dtype))

        return deltas


class Qwen3_5ForConditionalGenerationForPanoVLN(Qwen3_5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = list(
        getattr(Qwen3_5ForConditionalGeneration, "_keys_to_ignore_on_load_unexpected", []) or []
    ) + [r"panovggt\..*", r"paqr\..*"]

    PANOVGGT_STATE_KEYS = (
        "panovggt_mlp.alpha_value",
        "panovggt_mlp.input_norm.weight",
        "panovggt_mlp.mlp.0.weight",
        "panovggt_mlp.mlp.0.bias",
        "panovggt_mlp.mlp.2.weight",
        "panovggt_mlp.mlp.2.bias",
        "panovggt_mlp.output_norm.weight",
    )
    ACTION_BEARING_STATE_KEYS = (
        "action_bearing_residual.bin_embeddings",
        "action_bearing_residual.input_norm.weight",
        "action_bearing_residual.mlp.0.weight",
        "action_bearing_residual.mlp.0.bias",
        "action_bearing_residual.mlp.2.weight",
        "action_bearing_residual.mlp.2.bias",
        "action_bearing_residual.output_norm.weight",
        "action_bearing_residual.raw_alpha",
    )
    PAQR_STATE_KEYS = (
        "paqr.raw_gate",
        "paqr.raw_turn_cos",
        "paqr.raw_turn_sin",
        "paqr.raw_forward_cos",
    )

    def __init__(self, config):
        panovggt_enabled = bool(getattr(config, "panovggt_enabled", False))
        action_bearing_enabled = bool(
            getattr(config, "action_bearing_enabled", False)
        )
        paqr_enabled = bool(getattr(config, "paqr_enabled", False))
        if action_bearing_enabled and paqr_enabled:
            raise ValueError(
                "action_bearing_enabled and paqr_enabled cannot both be true"
            )
        if panovggt_enabled:
            ensure_panovggt_config(config)
        if action_bearing_enabled:
            ensure_action_bearing_config(config)
        if paqr_enabled:
            ensure_paqr_config(config)
        super().__init__(config)

        if panovggt_enabled:
            ensure_panovggt_config(config)
        if action_bearing_enabled:
            ensure_action_bearing_config(config)
        if paqr_enabled:
            ensure_paqr_config(config)

        self.panovggt_mlp = PanoVGGTGeometryMLP(config) if panovggt_enabled else None
        self.action_bearing_residual = (
            ActionBearingResidual(config)
            if action_bearing_enabled
            else None
        )
        self.paqr = PanoramicActionQueryReadout(config) if paqr_enabled else None
        if self.action_bearing_residual is not None:
            self.action_bearing_residual.initialize_bin_embeddings_from_text(
                self.get_input_embeddings()
            )
        self.panovggt = None
        self._panovggt_weights_ready = False
        self._panovggt_dtype = None
        self._panovggt_device = None
        self._panovggt_mlp_load_was_incompatible = False
        self._action_bearing_initialized_after_load = False
        self._paqr_initialized_after_load = False
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None
        self._pano_runtime_in_image_features = False
        self._pano_runtime_visual_grid_thw = None
        self._install_pano_merger_hook()
        self._install_image_feature_hook()
        self.register_load_state_dict_pre_hook(self._drop_incompatible_panovggt_mlp_pre_hook)
        self.register_load_state_dict_post_hook(self._load_missing_pano_parameters_post_hook)

    def _action_bearing_enabled(self) -> bool:
        return (
            self.action_bearing_residual is not None
            and bool(getattr(self.action_bearing_residual, "enabled", False))
        )

    def _paqr_enabled(self) -> bool:
        return self.paqr is not None and bool(getattr(self.paqr, "enabled", False))

    def _panovggt_enabled(self) -> bool:
        return self.panovggt_mlp is not None and bool(getattr(self.panovggt_mlp, "enabled", False))

    def _panovggt_feature_source(self) -> str:
        if self.panovggt_mlp is not None:
            return self.panovggt_mlp.feature_source
        return normalize_panovggt_feature_source(getattr(self.config, "panovggt_feature_source", "aggregator"))

    def _panovggt_injection_stage(self) -> str:
        if self.panovggt_mlp is not None:
            return self.panovggt_mlp.injection_stage
        return normalize_panovggt_injection_stage(getattr(self.config, "panovggt_injection_stage", "post_merger"))

    def _runtime_grid_matches_image_grid(self, grid_thw: torch.Tensor | None) -> bool:
        runtime_grid = self._pano_runtime_grid_thw
        if grid_thw is None or runtime_grid is None:
            return False
        if grid_thw is runtime_grid:
            return True
        if grid_thw.device == runtime_grid.device and tuple(grid_thw.shape) == tuple(runtime_grid.shape):
            return grid_thw.data_ptr() == runtime_grid.data_ptr()
        return False

    def _select_current_items(
        self,
        image_grid_thw: torch.Tensor,
        num_image_outputs: int | None = None,
    ) -> list[tuple[int, int]]:
        current_indices = self._current_image_flat_indices(image_grid_thw)
        if not current_indices:
            return []
        return [
            (batch_index, image_index)
            for batch_index, image_index in current_indices
            if 0 <= image_index < int(image_grid_thw.shape[0])
            and (num_image_outputs is None or image_index < num_image_outputs)
        ]

    def _panovggt_current_batch_inputs(
        self,
        *,
        panovggt_pixel_values: torch.Tensor,
        current_items: list[tuple[int, int]],
    ) -> torch.Tensor:
        if panovggt_pixel_values is None or panovggt_pixel_values.numel() == 0:
            raise AssertionError("PanoVGGT is enabled but panovggt_pixel_values is missing")
        if int(panovggt_pixel_values.shape[0]) != len(current_items):
            raise AssertionError(
                "PanoVGGT current-image batch size mismatch: "
                f"panovggt_batch={int(panovggt_pixel_values.shape[0])}, "
                f"current_items={len(current_items)}"
            )
        batch_indices = torch.tensor(
            [item[0] for item in current_items],
            device=panovggt_pixel_values.device,
            dtype=torch.long,
        )
        return panovggt_pixel_values.index_select(0, batch_indices)

    def _target_geometry_for_indices(
        self,
        target_indices: list[int],
    ) -> torch.Tensor | None:
        geometry = self._pano_runtime_image_geometry
        if geometry is None:
            return None
        return geometry[target_indices].detach().to(device="cpu", dtype=torch.float32)

    def _apply_post_merger_panovggt_residual(self, vision_output, image_grid_thw: torch.Tensor):
        if image_grid_thw is None or vision_output.pooler_output is None:
            return vision_output

        image_embeds = list(vision_output.pooler_output)
        current_items = self._select_current_items(
            image_grid_thw,
            num_image_outputs=len(image_embeds),
        )
        if not current_items:
            return vision_output

        panovggt_pixel_values = self._pano_runtime_panovggt_pixel_values
        panovggt_inputs = self._panovggt_current_batch_inputs(
            panovggt_pixel_values=panovggt_pixel_values,
            current_items=current_items,
        )
        target_indices = [item[1] for item in current_items]
        target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
        target_lengths = [int(image_embeds[index].shape[0]) for index in target_indices]
        geometry = self._target_geometry_for_indices(target_indices)

        panovggt_model = self._ensure_panovggt_model(
            device=image_embeds[target_indices[0]].device,
            dtype=image_embeds[target_indices[0]].dtype,
        )
        if panovggt_model is None or self.panovggt_mlp is None:
            return vision_output
        deltas = self.panovggt_mlp(
            panovggt_inputs,
            panovggt_model=panovggt_model,
            target_grid_thw=target_grid_thw,
            target_lengths=target_lengths,
            image_erp_geometry=geometry,
            output_device=image_embeds[target_indices[0]].device,
            output_dtype=image_embeds[target_indices[0]].dtype,
        )
        for image_index, delta in zip(target_indices, deltas):
            if delta.shape != image_embeds[image_index].shape:
                raise AssertionError(
                    "PanoVGGT delta shape mismatch: "
                    f"delta={tuple(delta.shape)}, qwen={tuple(image_embeds[image_index].shape)}"
                )
            image_embeds[image_index] = image_embeds[image_index] + delta

        vision_output.pooler_output = tuple(image_embeds)
        return vision_output

    def _apply_action_bearing_residual(self, vision_output, image_grid_thw: torch.Tensor):
        action_bearing_residual = self.action_bearing_residual
        if (
            action_bearing_residual is None
            or image_grid_thw is None
            or vision_output.pooler_output is None
        ):
            return vision_output

        image_embeds = list(vision_output.pooler_output)
        current_items = self._select_current_items(
            image_grid_thw,
            num_image_outputs=len(image_embeds),
        )
        for _, image_index in current_items:
            image_embeds[image_index] = action_bearing_residual(
                image_embeds[image_index],
                image_grid_thw[image_index],
            )
        vision_output.pooler_output = tuple(image_embeds)
        return vision_output

    def _apply_pre_merger_panovggt_residual(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_grid_thw is None or hidden_states.numel() == 0:
            return hidden_states

        current_items = self._select_current_items(image_grid_thw)
        if not current_items:
            return hidden_states

        panovggt_pixel_values = self._pano_runtime_panovggt_pixel_values
        panovggt_inputs = self._panovggt_current_batch_inputs(
            panovggt_pixel_values=panovggt_pixel_values,
            current_items=current_items,
        )
        target_indices = [item[1] for item in current_items]
        target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
        target_lengths = [
            int(num_frames * grid_h * grid_w)
            for num_frames, grid_h, grid_w in target_grid_thw.tolist()
        ]
        geometry = self._target_geometry_for_indices(target_indices)

        panovggt_model = self._ensure_panovggt_model(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if panovggt_model is None or self.panovggt_mlp is None:
            return hidden_states
        deltas = self.panovggt_mlp(
            panovggt_inputs,
            panovggt_model=panovggt_model,
            target_grid_thw=target_grid_thw,
            target_lengths=target_lengths,
            image_erp_geometry=geometry,
            output_device=hidden_states.device,
            output_dtype=hidden_states.dtype,
        )

        split_lengths = [
            int(num_frames * grid_h * grid_w)
            for num_frames, grid_h, grid_w in image_grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        offsets = []
        offset = 0
        for length in split_lengths:
            offsets.append(offset)
            offset += length
        if offset != int(hidden_states.shape[0]):
            raise AssertionError(
                "Qwen pre-merger hidden length does not match image_grid_thw: "
                f"hidden={int(hidden_states.shape[0])}, grid_total={offset}"
            )

        hidden_states = hidden_states.clone()
        for image_index, delta in zip(target_indices, deltas):
            start = offsets[image_index]
            length = split_lengths[image_index]
            if tuple(delta.shape) != (length, int(hidden_states.shape[-1])):
                raise AssertionError(
                    "PanoVGGT pre-merger delta shape mismatch: "
                    f"delta={tuple(delta.shape)}, qwen=({length}, {int(hidden_states.shape[-1])})"
                )
            hidden_states[start:start + length] = hidden_states[start:start + length] + delta
        return hidden_states

    def _install_pano_merger_hook(self) -> None:
        if not self._panovggt_enabled() or self._panovggt_injection_stage() != "pre_merger":
            return
        visual = self.model.visual
        if hasattr(visual.merger, "_pano_origin_forward"):
            return

        visual.merger._pano_origin_forward = visual.merger.forward
        owner = self

        def merger_with_pano_pre_merger_residual(this, hidden_states):
            if (
                owner._panovggt_enabled()
                and owner._panovggt_injection_stage() == "pre_merger"
                and owner._pano_runtime_in_image_features
            ):
                hidden_states = owner._apply_pre_merger_panovggt_residual(
                    hidden_states=hidden_states,
                    image_grid_thw=owner._pano_runtime_visual_grid_thw,
                )
            return this._pano_origin_forward(hidden_states)

        visual.merger.forward = MethodType(merger_with_pano_pre_merger_residual, visual.merger)

    def _install_image_feature_hook(self) -> None:
        if not (self._panovggt_enabled() or self._action_bearing_enabled()):
            return
        if hasattr(self.model, "_pano_origin_get_image_features"):
            return

        self.model._pano_origin_get_image_features = self.model.get_image_features
        owner = self

        def get_image_features_with_pano_residuals(this, pixel_values, image_grid_thw=None, **kwargs):
            is_image_visual_call = owner._runtime_grid_matches_image_grid(image_grid_thw)
            previous_in_image_features = owner._pano_runtime_in_image_features
            previous_visual_grid = owner._pano_runtime_visual_grid_thw
            owner._pano_runtime_in_image_features = is_image_visual_call
            owner._pano_runtime_visual_grid_thw = image_grid_thw if is_image_visual_call else None
            try:
                vision_output = this._pano_origin_get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kwargs,
                )
            finally:
                owner._pano_runtime_in_image_features = previous_in_image_features
                owner._pano_runtime_visual_grid_thw = previous_visual_grid
            if not is_image_visual_call:
                return vision_output
            if (
                owner._panovggt_enabled()
                and owner._panovggt_injection_stage() == "post_merger"
            ):
                vision_output = owner._apply_post_merger_panovggt_residual(
                    vision_output,
                    image_grid_thw,
                )
            if owner._action_bearing_enabled():
                vision_output = owner._apply_action_bearing_residual(
                    vision_output,
                    image_grid_thw,
                )
            return vision_output

        self.model.get_image_features = MethodType(get_image_features_with_pano_residuals, self.model)

    def _set_runtime_pano_context(
        self,
        *,
        image_grid_thw: torch.Tensor | None,
        image_num_images: torch.Tensor | None,
        image_current_index: torch.Tensor | None,
        image_erp_geometry: torch.Tensor | None,
        panovggt_pixel_values: torch.Tensor | None,
    ) -> None:
        self._pano_runtime_grid_thw = image_grid_thw
        self._pano_runtime_image_num_images = image_num_images
        self._pano_runtime_image_current_index = image_current_index
        self._pano_runtime_image_geometry = image_erp_geometry
        self._pano_runtime_panovggt_pixel_values = panovggt_pixel_values

    def _clear_runtime_pano_context(self) -> None:
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None
        self._pano_runtime_in_image_features = False
        self._pano_runtime_visual_grid_thw = None

    def _drop_incompatible_panovggt_mlp_pre_hook(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        if self.panovggt_mlp is None:
            return
        key_prefix = f"{prefix}panovggt_mlp."
        checkpoint_keys = [key for key in state_dict.keys() if key.startswith(key_prefix)]
        if not checkpoint_keys:
            return
        target_state = self.panovggt_mlp.state_dict()
        incompatible = False
        unknown_keys = []
        for key in checkpoint_keys:
            local_key = key[len(key_prefix) :]
            target_tensor = target_state.get(local_key)
            source_tensor = state_dict[key]
            if target_tensor is None:
                unknown_keys.append(key)
                continue
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                incompatible = True
                break
        if not incompatible:
            for key in unknown_keys:
                del state_dict[key]
            return
        for key in checkpoint_keys:
            del state_dict[key]
        self._panovggt_mlp_load_was_incompatible = True

    def _load_external_panovggt_weights(self) -> None:
        if not self._panovggt_enabled():
            return
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config(self._panovggt_feature_source())
        load_panovggt_checkpoint(
            self.panovggt,
            str(getattr(self.config, "panovggt_checkpoint_path")),
        )
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None

    @staticmethod
    def _load_prefixed_checkpoint_tensors(checkpoint_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return {}

        state_dict: dict[str, torch.Tensor] = {}

        def add_tensor(key: str, tensor: torch.Tensor) -> None:
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = tensor

        safetensors_index = checkpoint_dir / "model.safetensors.index.json"
        if safetensors_index.exists():
            from safetensors import safe_open

            weight_map = json.loads(safetensors_index.read_text()).get("weight_map", {})
            files_to_keys: dict[str, list[str]] = {}
            for key, filename in weight_map.items():
                if key.startswith(prefix):
                    files_to_keys.setdefault(filename, []).append(key)
            for filename, keys in files_to_keys.items():
                with safe_open(str(checkpoint_dir / filename), framework="pt", device="cpu") as handle:
                    for key in keys:
                        add_tensor(key, handle.get_tensor(key))
            return state_dict

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    add_tensor(key, handle.get_tensor(key))
            return state_dict

        def add_from_torch_file(path: Path, keys: set[str] | None = None) -> None:
            loaded = torch.load(str(path), map_location="cpu")
            if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                loaded = loaded["state_dict"]
            if not isinstance(loaded, dict):
                return
            for key, tensor in loaded.items():
                if keys is not None and key not in keys:
                    continue
                add_tensor(key, tensor)

        torch_index = checkpoint_dir / "pytorch_model.bin.index.json"
        if torch_index.exists():
            weight_map = json.loads(torch_index.read_text()).get("weight_map", {})
            files_to_keys: dict[str, set[str]] = {}
            for key, filename in weight_map.items():
                if key.startswith(prefix):
                    files_to_keys.setdefault(filename, set()).add(key)
            for filename, keys in files_to_keys.items():
                add_from_torch_file(checkpoint_dir / filename, keys)
            return state_dict

        torch_path = checkpoint_dir / "pytorch_model.bin"
        if torch_path.exists():
            add_from_torch_file(torch_path)

        return state_dict

    def _load_saved_panovggt_weights(self, pretrained_model_name_or_path) -> bool:
        if not self._panovggt_enabled():
            return False

        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        state_dict = self._load_prefixed_checkpoint_tensors(checkpoint_dir, "panovggt.")
        if not state_dict:
            return False

        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config(self._panovggt_feature_source())

        incompatible = self.panovggt.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            self.panovggt = None
            self._panovggt_weights_ready = False
            return False
        freeze_panovggt_model(self.panovggt)
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None
        return True

    def _mark_panovggt_weights_ready(self) -> None:
        if self.panovggt is None:
            return
        freeze_panovggt_model(self.panovggt)
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None

    def _ensure_panovggt_model(self, device: torch.device, dtype: torch.dtype):
        if not self._panovggt_enabled():
            return None
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config(self._panovggt_feature_source())
        if not self._panovggt_weights_ready:
            self._load_external_panovggt_weights()
        if bool(getattr(self.config, "panovggt_force_fp32", False)):
            target_dtype = torch.float32
        else:
            # Keep the frozen encoder in Qwen's low-precision vision dtype when
            # possible. PanoVGGT's attention has a bf16 flash path; forcing fp32
            # makes its 36-layer panorama encoder much slower.
            target_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
        if self._panovggt_device != device or self._panovggt_dtype != target_dtype:
            self.panovggt.to(device=device, dtype=target_dtype)
            self._panovggt_device = device
            self._panovggt_dtype = target_dtype
        freeze_panovggt_model(self.panovggt)
        return self.panovggt

    def _current_image_flat_indices(self, image_grid_thw: torch.Tensor | None) -> list[tuple[int, int]]:
        if image_grid_thw is None or image_grid_thw.numel() == 0:
            return []

        num_images = int(image_grid_thw.shape[0])
        device = image_grid_thw.device
        image_num_images = self._pano_runtime_image_num_images
        if image_num_images is None:
            image_num_images = torch.tensor([num_images], device=device, dtype=torch.long)
        else:
            image_num_images = image_num_images.to(device=device, dtype=torch.long)

        if int(image_num_images.sum().item()) != num_images:
            raise AssertionError(
                "image_num_images does not sum to image_grid_thw rows: "
                f"counts={image_num_images.tolist()}, grids={num_images}"
            )

        image_current_index = self._pano_runtime_image_current_index
        if image_current_index is None:
            image_current_index = image_num_images - 1
        else:
            image_current_index = image_current_index.to(device=device, dtype=torch.long)

        offsets = image_num_images.cumsum(0) - image_num_images
        valid = image_num_images > 0
        if torch.any(valid):
            invalid = (image_current_index < 0) | (image_current_index >= image_num_images)
            if torch.any(invalid & valid):
                raise AssertionError(
                    "image_current_index is out of range for at least one sample: "
                    f"indices={image_current_index.tolist()}, counts={image_num_images.tolist()}"
                )

        flat_indices = offsets + image_current_index
        return [
            (batch_index, int(flat_indices[batch_index].item()))
            for batch_index in range(int(image_num_images.shape[0]))
            if bool(valid[batch_index].item())
        ]

    def _merged_image_grid_shape(
        self,
        grid_thw: torch.Tensor,
        *,
        require_single_frame: bool = False,
    ) -> tuple[int, int, int]:
        num_frames, grid_h, grid_w = [
            int(value)
            for value in grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        merge_size = int(getattr(self.config.vision_config, "spatial_merge_size", 2))
        if num_frames < 1 or grid_h < 1 or grid_w < 1:
            raise AssertionError(
                f"Invalid image_grid_thw row for PAQR: {[num_frames, grid_h, grid_w]}"
            )
        if require_single_frame and num_frames != 1:
            raise AssertionError(
                "PAQR expects the current panorama to be one image frame, got "
                f"image_grid_thw={[num_frames, grid_h, grid_w]}"
            )
        if grid_h % merge_size != 0 or grid_w % merge_size != 0:
            raise AssertionError(
                "PAQR image grid is not divisible by spatial_merge_size: "
                f"image_grid_thw={[num_frames, grid_h, grid_w]}, "
                f"spatial_merge_size={merge_size}"
            )
        target_h = grid_h // merge_size
        target_w = grid_w // merge_size
        return num_frames * target_h * target_w, target_h, target_w

    def _current_panorama_hidden_tokens(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor | None,
        image_grid_thw: torch.LongTensor | None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if input_ids is None:
            raise AssertionError("PAQR requires input_ids to locate current-panorama tokens")
        if image_grid_thw is None or image_grid_thw.numel() == 0:
            raise AssertionError("PAQR requires image_grid_thw")
        if input_ids.ndim != 2 or hidden_states.ndim != 3:
            raise AssertionError(
                "Unexpected PAQR input shapes: "
                f"input_ids={tuple(input_ids.shape)}, hidden={tuple(hidden_states.shape)}"
            )
        if tuple(input_ids.shape) != tuple(hidden_states.shape[:2]):
            raise AssertionError(
                "PAQR input_ids/hidden sequence mismatch: "
                f"input_ids={tuple(input_ids.shape)}, hidden={tuple(hidden_states.shape)}"
            )

        num_images = int(image_grid_thw.shape[0])
        image_num_images = self._pano_runtime_image_num_images
        if image_num_images is None:
            image_num_images = torch.tensor(
                [num_images],
                device=image_grid_thw.device,
                dtype=torch.long,
            )
        else:
            image_num_images = image_num_images.to(
                device=image_grid_thw.device,
                dtype=torch.long,
            )
        if int(image_num_images.numel()) != int(input_ids.shape[0]):
            raise AssertionError(
                "PAQR image_num_images batch mismatch: "
                f"counts={image_num_images.tolist()}, batch={input_ids.shape[0]}"
            )
        if int(image_num_images.sum().item()) != num_images:
            raise AssertionError(
                "PAQR image_num_images does not sum to image_grid_thw rows: "
                f"counts={image_num_images.tolist()}, grids={num_images}"
            )

        current_by_batch = dict(self._current_image_flat_indices(image_grid_thw))
        image_token_id = int(getattr(self.config, "image_token_id"))
        panorama_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        grid_offset = 0
        for batch_index, count_tensor in enumerate(image_num_images):
            image_count = int(count_tensor.item())
            if image_count <= 0:
                continue
            sample_grids = image_grid_thw[grid_offset : grid_offset + image_count]
            image_lengths = [
                self._merged_image_grid_shape(grid_row)[0]
                for grid_row in sample_grids
            ]
            image_positions = torch.nonzero(
                input_ids[batch_index].eq(image_token_id),
                as_tuple=False,
            ).flatten()
            expected_positions = sum(image_lengths)
            if int(image_positions.numel()) != expected_positions:
                raise AssertionError(
                    "PAQR image placeholder count does not match merged image grids: "
                    f"sample={batch_index}, placeholders={image_positions.numel()}, "
                    f"expected={expected_positions}, image_lengths={image_lengths}"
                )

            current_flat_index = current_by_batch.get(batch_index)
            if current_flat_index is None:
                grid_offset += image_count
                continue
            current_local_index = current_flat_index - grid_offset
            token_start = sum(image_lengths[:current_local_index])
            token_end = token_start + image_lengths[current_local_index]
            current_positions = image_positions[token_start:token_end]
            current_hidden = hidden_states[batch_index].index_select(
                0,
                current_positions.to(device=hidden_states.device),
            )

            _, target_h, target_w = self._merged_image_grid_shape(
                sample_grids[current_local_index],
                require_single_frame=True,
            )
            yaw_columns = (
                (
                    torch.arange(
                        target_w,
                        device=current_hidden.device,
                        dtype=torch.float32,
                    )
                    + 0.5
                )
                / float(target_w)
                - 0.5
            ) * (2.0 * math.pi)
            yaw = yaw_columns.unsqueeze(0).expand(target_h, target_w).reshape(-1)
            if int(yaw.numel()) != int(current_hidden.shape[0]):
                raise AssertionError(
                    "PAQR current panorama yaw/token mismatch: "
                    f"yaw={yaw.numel()}, tokens={current_hidden.shape[0]}"
                )
            panorama_tokens[batch_index] = (current_hidden, yaw)
            grid_offset += image_count

        return panorama_tokens

    def _paqr_decision_positions(
        self,
        *,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
    ) -> dict[int, int]:
        decision_positions: dict[int, int] = {}
        if labels is not None:
            allowed_targets = set(
                int(token_id)
                for token_id in self.paqr.action_token_ids.detach().cpu().tolist()
            )
            allowed_targets.add(int(getattr(self.config, "paqr_stop_token_id")))
            for batch_index in range(int(labels.shape[0])):
                supervised_positions = torch.nonzero(
                    labels[batch_index].ne(-100),
                    as_tuple=False,
                ).flatten()
                if supervised_positions.numel() == 0:
                    continue
                first_target_position = int(supervised_positions[0].item())
                if first_target_position <= 0:
                    raise AssertionError(
                        "PAQR first supervised token has no preceding decision position"
                    )
                first_target_id = int(
                    labels[batch_index, first_target_position].item()
                )
                if first_target_id not in allowed_targets:
                    raise AssertionError(
                        "PAQR expects the first target token to be one of "
                        f"{sorted(allowed_targets)}, got {first_target_id} "
                        f"for sample {batch_index}"
                    )
                decision_positions[batch_index] = first_target_position - 1
            return decision_positions

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        for batch_index in range(int(input_ids.shape[0])):
            active_positions = torch.nonzero(
                attention_mask[batch_index].ne(0),
                as_tuple=False,
            ).flatten()
            if active_positions.numel() > 0:
                decision_positions[batch_index] = int(active_positions[-1].item())
        return decision_positions

    @staticmethod
    def _logit_source_positions(
        sequence_length: int,
        logits_to_keep: int | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError(
                    f"logits_to_keep must be non-negative, got {logits_to_keep}"
                )
            start = (
                0
                if logits_to_keep == 0
                else max(0, sequence_length - logits_to_keep)
            )
            return torch.arange(start, sequence_length, device=device)

        positions = logits_to_keep.to(device=device, dtype=torch.long).flatten()
        positions = torch.where(
            positions < 0,
            positions + int(sequence_length),
            positions,
        )
        if torch.any((positions < 0) | (positions >= int(sequence_length))):
            raise IndexError(
                "logits_to_keep contains a position outside the hidden sequence"
            )
        return positions

    def _apply_paqr_to_logits(
        self,
        *,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.LongTensor,
        logits_to_keep: int | torch.Tensor,
    ) -> torch.Tensor:
        panorama_by_batch = self._current_panorama_hidden_tokens(
            hidden_states=hidden_states,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )
        decision_by_batch = self._paqr_decision_positions(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        source_positions = self._logit_source_positions(
            sequence_length=int(hidden_states.shape[1]),
            logits_to_keep=logits_to_keep,
            device=hidden_states.device,
        )
        action_token_ids = self.paqr.action_token_ids.to(device=logits.device)

        for batch_index, (panorama_hidden, yaw) in panorama_by_batch.items():
            decision_position = decision_by_batch.get(batch_index)
            if decision_position is None:
                continue
            matching_logit_positions = torch.nonzero(
                source_positions.eq(decision_position),
                as_tuple=False,
            ).flatten()
            if matching_logit_positions.numel() != 1:
                raise AssertionError(
                    "PAQR decision position is absent or duplicated in logits_to_keep: "
                    f"decision={decision_position}, source_positions="
                    f"{source_positions.detach().cpu().tolist()}"
                )
            logit_position = int(matching_logit_positions[0].item())
            delta = self.paqr(
                panorama_hidden=panorama_hidden,
                yaw=yaw,
                lm_head_weight=self.lm_head.weight,
            ).to(device=logits.device, dtype=logits.dtype)
            current_action_logits = logits[
                batch_index,
                logit_position,
            ].index_select(0, action_token_ids)
            logits[batch_index, logit_position].index_copy_(
                0,
                action_token_ids,
                current_action_logits + delta,
            )
        return logits

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_erp_geometry: torch.FloatTensor | None = None,
        image_num_images: torch.LongTensor | None = None,
        image_current_index: torch.LongTensor | None = None,
        panovggt_pixel_values: torch.Tensor | None = None,
        paqr_apply_s1: bool | None = None,
        **kwargs,
    ) -> Qwen3_5CausalLMOutputWithPast:
        runtime_context_needed = (
            self._panovggt_enabled()
            or self._action_bearing_enabled()
            or self._paqr_enabled()
        )
        if runtime_context_needed:
            self._set_runtime_pano_context(
                image_grid_thw=image_grid_thw,
                image_num_images=image_num_images,
                image_current_index=image_current_index,
                image_erp_geometry=image_erp_geometry,
                panovggt_pixel_values=panovggt_pixel_values,
            )
        try:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
            hidden_states = outputs[0]
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            should_apply_paqr = self._paqr_enabled() and (
                labels is not None or bool(paqr_apply_s1)
            )
            if should_apply_paqr:
                if input_ids is None or image_grid_thw is None:
                    raise AssertionError(
                        "PAQR first-action readout requires input_ids and image_grid_thw"
                    )
                logits = self._apply_paqr_to_logits(
                    logits=logits,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    image_grid_thw=image_grid_thw,
                    logits_to_keep=logits_to_keep,
                )

            loss = None
            if labels is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.text_config.vocab_size,
                )

            return Qwen3_5CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )
        finally:
            self._clear_runtime_pano_context()

    def _load_missing_pano_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)

        paqr_module = self.paqr
        if (
            paqr_module is not None
            and paqr_module.enabled
            and any(key.startswith("paqr.") for key in missing_keys)
        ):
            paqr_module.reset_parameters()
            self._paqr_initialized_after_load = True

        action_bearing_module = self.action_bearing_residual
        if (
            action_bearing_module is not None
            and action_bearing_module.enabled
            and any(key.startswith("action_bearing_residual.") for key in missing_keys)
        ):
            action_bearing_module.reset_parameters(
                float(getattr(self.config.vision_config, "initializer_range", 0.02))
            )
            action_bearing_module.initialize_bin_embeddings_from_text(
                self.get_input_embeddings()
            )
            self._action_bearing_initialized_after_load = True

        panovggt_module = self.panovggt_mlp
        if (
            panovggt_module is not None
            and panovggt_module.enabled
            and (
                self._panovggt_mlp_load_was_incompatible
                or any(key.startswith("panovggt_mlp.") for key in missing_keys)
            )
        ):
            panovggt_module.reset_parameters()
            self._panovggt_mlp_load_was_incompatible = False
        if panovggt_module is not None and panovggt_module.enabled:
            if any(key.startswith("panovggt.") for key in missing_keys):
                self._load_external_panovggt_weights()
            else:
                self._mark_panovggt_weights_ready()

    def _reset_action_bearing_parameters_after_pretrained_load(self) -> None:
        if self._action_bearing_enabled() and self.action_bearing_residual is not None:
            self.action_bearing_residual.reset_parameters(
                float(getattr(self.config.vision_config, "initializer_range", 0.02))
            )
            self.action_bearing_residual.initialize_bin_embeddings_from_text(
                self.get_input_embeddings()
            )
            self._action_bearing_initialized_after_load = True

    def _reset_paqr_parameters_after_pretrained_load(self) -> None:
        if self._paqr_enabled() and self.paqr is not None:
            self.paqr.reset_parameters()
            self._paqr_initialized_after_load = True

    def _reset_panovggt_parameters_after_pretrained_load(self) -> None:
        if self._panovggt_enabled() and self.panovggt_mlp is not None:
            self.panovggt_mlp.reset_parameters()

    @classmethod
    def _checkpoint_has_panovggt_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.PANOVGGT_STATE_KEYS)

    @classmethod
    def _checkpoint_has_action_bearing_weights(
        cls,
        pretrained_model_name_or_path,
    ) -> bool | None:
        return cls._checkpoint_has_any_weights(
            pretrained_model_name_or_path,
            cls.ACTION_BEARING_STATE_KEYS,
        )

    @classmethod
    def _checkpoint_has_paqr_weights(
        cls,
        pretrained_model_name_or_path,
    ) -> bool | None:
        return cls._checkpoint_has_any_weights(
            pretrained_model_name_or_path,
            cls.PAQR_STATE_KEYS,
        )

    @classmethod
    def _checkpoint_has_any_weights(cls, pretrained_model_name_or_path, state_keys) -> bool | None:
        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        if not checkpoint_dir.exists():
            return None

        index_candidates = (
            checkpoint_dir / "model.safetensors.index.json",
            checkpoint_dir / "pytorch_model.bin.index.json",
        )
        for index_path in index_candidates:
            if not index_path.exists():
                continue
            try:
                weight_map = json.loads(index_path.read_text()).get("weight_map", {})
            except Exception:
                return None
            return all(key in weight_map for key in state_keys)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors import safe_open

                with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                    keys = set(handle.keys())
                return all(key in keys for key in state_keys)
            except Exception:
                return None

        torch_path = checkpoint_dir / "pytorch_model.bin"
        if torch_path.exists():
            try:
                loaded = torch.load(str(torch_path), map_location="cpu")
                if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                    loaded = loaded["state_dict"]
                if not isinstance(loaded, dict):
                    return None
                return all(key in loaded for key in state_keys)
            except Exception:
                return None

        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        has_paqr_weights = cls._checkpoint_has_paqr_weights(
            pretrained_model_name_or_path
        )
        if (
            model._paqr_enabled()
            and has_paqr_weights is not True
            and not model._paqr_initialized_after_load
        ):
            model._reset_paqr_parameters_after_pretrained_load()
        has_action_bearing_weights = cls._checkpoint_has_action_bearing_weights(
            pretrained_model_name_or_path
        )
        if (
            model._action_bearing_enabled()
            and has_action_bearing_weights is not True
            and not model._action_bearing_initialized_after_load
        ):
            model._reset_action_bearing_parameters_after_pretrained_load()
        has_panovggt_weights = cls._checkpoint_has_panovggt_weights(pretrained_model_name_or_path)
        if has_panovggt_weights is False:
            model._reset_panovggt_parameters_after_pretrained_load()
        if model._panovggt_enabled():
            loaded_saved_panovggt = (
                model._load_saved_panovggt_weights(pretrained_model_name_or_path)
                if has_panovggt_weights is True
                else False
            )
            if not loaded_saved_panovggt and model.panovggt is None:
                model._load_external_panovggt_weights()
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        image_erp_geometry=None,
        image_num_images=None,
        image_current_index=None,
        panovggt_pixel_values=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            image_erp_geometry=image_erp_geometry,
            image_num_images=image_num_images,
            image_current_index=image_current_index,
            panovggt_pixel_values=panovggt_pixel_values,
            **kwargs,
        )
        if self._paqr_enabled():
            model_inputs["paqr_apply_s1"] = bool(is_first_iteration)
        if not is_first_iteration and use_cache:
            model_inputs["image_erp_geometry"] = None
            model_inputs["image_num_images"] = None
            model_inputs["image_current_index"] = None
            model_inputs["panovggt_pixel_values"] = None
        return model_inputs

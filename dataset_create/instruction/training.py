"""Clean ERP observations aligned with the source trajectory's actions."""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def training_states(episode, states):
    actions = episode["action_ids"]
    if not actions or actions[-1] != 0 or actions.count(0) != 1 or len(states) != len(actions) + 1:
        raise ValueError("Training frames require one terminal STOP and action-aligned replay states")
    for key in ("position", "rotation_xyzw"):
        if not np.allclose(states[-2][key], states[-1][key], atol=1e-6):
            raise ValueError("STOP must preserve the final observation pose")
    # Frame i is the observation BEFORE action i, including the STOP observation.
    return states[:-1]


def erp_directory(root, episode_id):
    name = str(episode_id)
    if not name.isdecimal():
        raise ValueError("episode_id must be a nonnegative integer")
    return Path(root) / name


def validate_erp(root, episode_id, count, settings, inspect_images=False):
    directory = erp_directory(root, episode_id)
    expected = {f"frame_{i}.jpg" for i in range(count)}
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected:
        raise ValueError(f"Incomplete clean ERP sequence: {directory}")
    for name in expected:
        path = directory / name
        if path.stat().st_size == 0:
            raise ValueError(f"Empty ERP frame: {path}")
        if inspect_images:
            with Image.open(path) as image:
                if image.size != (settings["width"], settings["height"]) or image.mode != "RGB":
                    raise ValueError(f"Invalid ERP image: {path}")
                image.verify()
    return directory


def export_clean_erp(renderer, episode, episode_id, states, root, settings):
    frames = training_states(episode, states)
    root = Path(root)
    directory = erp_directory(root, episode_id)
    temporary = root / f".{episode_id}.partial"
    # A terminated worker may leave an unfinished sequence. It is never reused
    # as training data; the next attempt rebuilds it before atomic publication.
    if temporary.exists():
        shutil.rmtree(temporary)
    if directory.exists():
        validate_erp(root, episode_id, len(frames), settings, inspect_images=True)
        return {"directory": str(directory), "frames": len(frames)}
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for i, state in enumerate(frames):
            rgb = renderer.observe_erp(state)
            Image.fromarray(rgb).save(temporary / f"frame_{i}.jpg", quality=settings["jpeg_quality"],
                                      subsampling=settings["jpeg_subsampling"])
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_erp(root, episode_id, len(frames), settings)
    return {"directory": str(directory), "frames": len(frames)}

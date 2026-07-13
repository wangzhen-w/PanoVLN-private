"""Portable HM3D scene IDs and local asset resolution."""

from pathlib import Path, PurePosixPath


HM3D_NAMESPACE = "hm3d"


def public_scene_id(scene: Path, scene_root: str) -> str:
    """Return ``hm3d/{split}/{scene}/{file}`` for a local HM3D asset."""

    root = Path(scene_root).resolve()
    resolved = scene.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Scene is outside scene root: {resolved} not under {root}"
        ) from error
    if len(relative.parts) != 3:
        raise ValueError(
            f"HM3D scene must be nested as <split>/<scene>/<file>: {resolved}"
        )
    return f"{HM3D_NAMESPACE}/{relative.as_posix()}"


def local_scene_id(scene_id: str) -> str:
    """Strip the public HM3D namespace for a root that already points at HM3D."""

    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("scene_id must be a non-empty string")
    path = PurePosixPath(scene_id)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"scene_id must be a safe relative path: {scene_id}")
    parts = path.parts
    if parts and parts[0] == HM3D_NAMESPACE:
        parts = parts[1:]
    if len(parts) != 3:
        raise ValueError(
            "HM3D scene_id must be <namespace>/<split>/<scene>/<file> "
            f"or the legacy <split>/<scene>/<file>: {scene_id}"
        )
    return PurePosixPath(*parts).as_posix()


def validate_public_scene_id(scene_id: str) -> str:
    """Require the official ``hm3d/{split}/{scene}/{file}.basis.glb`` form."""

    path = PurePosixPath(scene_id)
    canonical = path.as_posix()
    if scene_id != canonical:
        raise ValueError(f"Public HM3D scene_id is not canonical: {scene_id}")
    if not path.parts or path.parts[0] != HM3D_NAMESPACE:
        raise ValueError(f"Public HM3D scene_id must start with hm3d/: {scene_id}")
    local = local_scene_id(scene_id)
    if not local.endswith(".basis.glb"):
        raise ValueError(f"Public HM3D scene_id must end with .basis.glb: {scene_id}")
    return canonical


def resolve_scene_path(scene_root: str, scene_id: str) -> Path:
    """Resolve a public or legacy scene ID beneath the configured HM3D root."""

    root = Path(scene_root).resolve()
    candidate = (root / local_scene_id(scene_id)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Scene ID escapes scene root: {scene_id}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"Scene asset does not exist: {candidate}")
    return candidate

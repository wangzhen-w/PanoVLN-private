"""Perspective RGB-D surface decals and agent-relative compass coordinates.

Paint only reconstructed visible, upward-facing surfaces close to the grounded
GT polyline. Unlike a 2D projected stroke, a decal cannot bridge an occluder or
float at the navmesh's offset above the physical floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from dataset_create.instruction.segmentation import rotation_matrix


# Positive bearing is to the agent's RIGHT. All views share one camera origin.
COMPASS = (
    ("front_left", -45., 0, 0), ("front", 0., 0, 1), ("front_right", 45., 0, 2),
    ("left", -90., 1, 0), ("right", 90., 1, 2),
    ("back_left", -135., 2, 0), ("back", 180., 2, 1), ("back_right", 135., 2, 2),
)


def look_rotation(base_xyzw, bearing_degrees):
    # A negative world-Y rotation looks right in Habitat's local -Z convention.
    x, y, z, w = np.asarray(base_xyzw, dtype=float)
    half = math.radians(-bearing_degrees) / 2
    sy, cy = math.sin(half), math.cos(half)
    return [x*cy-z*sy, w*sy+y*cy, x*sy+z*cy, w*cy-y*sy]


@dataclass
class Camera:
    width: int
    height: int
    hfov_degrees: float
    position: np.ndarray
    rotation_xyzw: list[float]

    @property
    def focal(self):
        return self.width / (2 * math.tan(math.radians(self.hfov_degrees) / 2))

    def project(self, points):
        points = np.atleast_2d(points).astype(float)
        local = (points - self.position) @ rotation_matrix(self.rotation_xyzw)
        z = -local[:, 2]
        denominator = np.maximum(z, 1e-8)
        return np.c_[self.width / 2 - .5 + self.focal * local[:, 0] / denominator,
                     self.height / 2 - .5 - self.focal * local[:, 1] / denominator], z

    def unproject(self, depth):
        rows, cols = np.indices(depth.shape)
        local = np.stack(((cols + .5 - self.width/2) * depth / self.focal,
                          -(rows + .5 - self.height/2) * depth / self.focal,
                          -depth), axis=-1)
        return local @ rotation_matrix(self.rotation_xyzw).T + self.position


def surface_normals(points):
    # Normals at discontinuities are unreliable: mark large depth jumps invalid.
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1)
    up = np.abs(normals[..., 1]) / np.maximum(norm, 1e-12)
    valid = ((norm > 1e-10) & (np.linalg.norm(dx, axis=-1) < .35) &
             (np.linalg.norm(dy, axis=-1) < .35))
    return up, valid


def densify_positions(positions, spacing):
    """Return real action-polyline samples, tangent and arclength, including bends."""
    positions = np.asarray(positions, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(positions, axis=0), axis=1) > 1e-5]
    positions = positions[keep]
    if len(positions) < 2:
        return positions, np.zeros_like(positions), np.zeros(len(positions))
    points, tangents, arcs = [], [], []
    arc = 0.
    for start, end in zip(positions[:-1], positions[1:]):
        vector = end - start
        length = float(np.linalg.norm(vector))
        tangent = vector.copy()
        tangent[1] = 0
        tangent /= max(np.linalg.norm(tangent), 1e-8)
        n = max(1, int(math.ceil(length / spacing)))
        for alpha in np.arange(n) / n:
            points.append(start + alpha * vector)
            tangents.append(tangent)
            arcs.append(arc + alpha * length)
        arc += length
    return (np.asarray(points + [positions[-1]]),
            np.asarray(tangents + [tangents[-1]]), np.asarray(arcs + [arc]))


@dataclass
class GroundRoute:
    points: np.ndarray
    tangents: np.ndarray
    arc: np.ndarray
    valid: np.ndarray

    def subset(self, start_m, end_m):
        mask = (self.arc >= start_m - .025) & (self.arc <= end_m + .025)
        return GroundRoute(self.points[mask], self.tangents[mask], self.arc[mask], self.valid[mask])


def overlay_route(rgb, depth, camera, route, settings):
    points = camera.unproject(depth)
    up, normals_valid = surface_normals(points)
    surface = ((depth > .05) & (depth < settings["far_m"] - .05) & np.isfinite(depth) &
               normals_valid & (up >= settings["floor_normal_min_y"]))
    route_valid = route.valid & np.isfinite(route.points).all(axis=1)
    mask = np.zeros(depth.shape, dtype=bool)
    if not route_valid.any() or not surface.any():
        return np.asarray(rgb)[..., :3].copy(), {"route_pixels": 0}
    valid_points = route.points[route_valid]
    flat = points[surface]
    # Full 3D lookup disambiguates stacked floors and routes that cross in x/z.
    distances, ids = cKDTree(valid_points).query(flat)
    nearest = valid_points[ids]
    tangent = route.tangents[route_valid][ids]
    delta = flat - nearest
    along = np.sum(delta * tangent, axis=1)
    lateral = np.abs(delta[:, 0] * tangent[:, 2] - delta[:, 2] * tangent[:, 0])
    s = route.arc[route_valid][ids] + along
    # The head points toward increasing GT arclength, with a broad trailing base.
    phase = np.mod(s, settings["arrow_spacing_m"])
    remaining = settings["arrow_length_m"] - phase
    head = ((remaining >= 0) &
            (lateral <= settings["arrow_width_m"] / 2 * remaining / settings["arrow_length_m"]))
    stripe = lateral <= settings["route_width_m"] / 2
    on_path = ((stripe | head) &
               (np.abs(delta[:, 1]) <= settings["surface_height_tolerance_m"]) &
               (distances <= max(settings["arrow_width_m"], settings["route_sample_m"] * 2)) &
               (s >= route.arc[0] - .02) & (s <= route.arc[-1] + .02))
    mask[surface] = on_path
    result = np.asarray(rgb)[..., :3].copy()
    result[mask] = (.15 * result[mask] + .85 * np.array([255, 170, 20])).astype(np.uint8)
    return result, {"route_pixels": int(mask.sum())}

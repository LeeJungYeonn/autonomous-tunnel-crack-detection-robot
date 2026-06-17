import argparse
import csv
import math
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = MODEL_DIR / "cracks/crack_texture_size_gt.csv"
DEFAULT_OUTPUT_CSV = MODEL_DIR / "gt/camera_coverage_report.csv"

TUNNEL_X_MIN_M = -5.0
TUNNEL_X_MAX_M = 5.0
TUNNEL_LENGTH_M = TUNNEL_X_MAX_M - TUNNEL_X_MIN_M
TUNNEL_INNER_RADIUS_M = 0.70

ROBOT_BASE_Z_M = 0.09
ROBOT_X_SAMPLE_STEP_M = 0.01

IMAGE_W = 640
IMAGE_H = 480
NEAR_CLIP_M = 0.05
FAR_CLIP_M = 1.3
EDGE_MARGIN_TARGET_PX = 8.0

CAMERA_OPTICAL_RPY = (-1.5708, 0.0, -1.5708)
CAMERAS = (
    {
        "name": "left",
        "xyz": (0.18, 0.17, 0.11),
        "rpy": (0.0, -0.28, 1.25),
        "horizontal_fov": 1.70,
    },
    {
        "name": "top",
        "xyz": (0.16, 0.0, 0.10),
        "rpy": (0.0, -1.5708, 0.0),
        "horizontal_fov": 1.92,
    },
    {
        "name": "right",
        "xyz": (0.18, -0.17, 0.11),
        "rpy": (0.0, -0.28, -1.25),
        "horizontal_fov": 1.70,
    },
)


def rotation_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def rotation_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def rotation_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_from_rpy(roll, pitch, yaw):
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def uv_to_world_point(texture_u, texture_v):
    theta = texture_u / 0.5 * math.pi
    world_x = TUNNEL_X_MIN_M + texture_v * TUNNEL_LENGTH_M
    world_y = TUNNEL_INNER_RADIUS_M * math.cos(theta)
    world_z = TUNNEL_INNER_RADIUS_M * math.sin(theta)
    return np.array([world_x, world_y, world_z], dtype=np.float64)


def build_camera_models():
    optical_rotation = rotation_from_rpy(*CAMERA_OPTICAL_RPY)
    models = []
    for camera in CAMERAS:
        rotation = rotation_from_rpy(*camera["rpy"]) @ optical_rotation
        horizontal_fov = camera["horizontal_fov"]
        vertical_fov = 2.0 * math.atan(
            math.tan(horizontal_fov / 2.0) * (IMAGE_H / IMAGE_W)
        )
        models.append(
            {
                "name": camera["name"],
                "xyz": np.array(camera["xyz"], dtype=np.float64),
                "rotation": rotation,
                "tan_h": math.tan(horizontal_fov / 2.0),
                "tan_v": math.tan(vertical_fov / 2.0),
            }
        )
    return models


def project_point(world_point, robot_x, camera):
    camera_world = np.array(
        [robot_x, 0.0, ROBOT_BASE_Z_M],
        dtype=np.float64
    ) + camera["xyz"]
    camera_point = camera["rotation"].T @ (world_point - camera_world)
    depth = camera_point[2]
    if depth <= NEAR_CLIP_M or depth >= FAR_CLIP_M:
        return None

    norm_x = camera_point[0] / depth
    norm_y = camera_point[1] / depth
    if abs(norm_x) > camera["tan_h"] or abs(norm_y) > camera["tan_v"]:
        return None

    pixel_x = (norm_x / camera["tan_h"] + 1.0) * 0.5 * (IMAGE_W - 1)
    pixel_y = (norm_y / camera["tan_v"] + 1.0) * 0.5 * (IMAGE_H - 1)
    return {
        "pixel_x": float(pixel_x),
        "pixel_y": float(pixel_y),
        "depth_m": float(depth),
    }


def score_candidate(center_projection, corner_projections):
    visible_corners = [p for p in corner_projections if p is not None]
    full_bbox_visible = len(visible_corners) == len(corner_projections)
    projected = [center_projection] + visible_corners
    margin_px = min(
        min(
            p["pixel_x"],
            IMAGE_W - 1 - p["pixel_x"],
            p["pixel_y"],
            IMAGE_H - 1 - p["pixel_y"],
        )
        for p in projected
    )

    bbox_w_px = 0.0
    bbox_h_px = 0.0
    bbox_diag_px = 0.0
    if full_bbox_visible:
        xs = np.array([p["pixel_x"] for p in corner_projections])
        ys = np.array([p["pixel_y"] for p in corner_projections])
        bbox_w_px = float(xs.max() - xs.min())
        bbox_h_px = float(ys.max() - ys.min())
        bbox_diag_px = float(math.hypot(bbox_w_px, bbox_h_px))

    return {
        "full_bbox_visible": full_bbox_visible,
        "visible_corner_count": len(visible_corners),
        "margin_px": float(margin_px),
        "bbox_w_px": bbox_w_px,
        "bbox_h_px": bbox_h_px,
        "bbox_diag_px": bbox_diag_px,
    }


def choose_better(current, candidate):
    if current is None:
        return candidate

    current_margin_score = min(current["margin_px"], EDGE_MARGIN_TARGET_PX)
    candidate_margin_score = min(candidate["margin_px"], EDGE_MARGIN_TARGET_PX)
    current_key = (
        int(current["full_bbox_visible"]),
        current["visible_corner_count"],
        current_margin_score,
        current["bbox_diag_px"],
        current["margin_px"],
        -abs(current["crack_world_x_m"] - current["camera_world_x_m"]),
    )
    candidate_key = (
        int(candidate["full_bbox_visible"]),
        candidate["visible_corner_count"],
        candidate_margin_score,
        candidate["bbox_diag_px"],
        candidate["margin_px"],
        -abs(candidate["crack_world_x_m"] - candidate["camera_world_x_m"]),
    )
    if candidate_key > current_key:
        return candidate
    return current


def evaluate_crack(row, cameras, robot_xs):
    u_min = float(row["u_min"])
    u_max = float(row["u_max"])
    v_min = float(row["v_min"])
    v_max = float(row["v_max"])
    center_u = float(row["target_u_center"])
    center_v = float(row["target_v_center"])

    center = uv_to_world_point(center_u, center_v)
    corners = [
        uv_to_world_point(u_min, v_min),
        uv_to_world_point(u_max, v_min),
        uv_to_world_point(u_max, v_max),
        uv_to_world_point(u_min, v_max),
    ]

    best = None
    for robot_x in robot_xs:
        for camera in cameras:
            center_projection = project_point(
                center,
                robot_x,
                camera
            )
            if center_projection is None:
                continue
            corner_projections = [
                project_point(corner, robot_x, camera)
                for corner in corners
            ]
            candidate = score_candidate(center_projection, corner_projections)
            candidate.update(
                {
                    "camera": camera["name"],
                    "robot_x_m": float(robot_x),
                    "camera_world_x_m": float(robot_x + camera["xyz"][0]),
                    "center_depth_m": center_projection["depth_m"],
                    "center_pixel_x": center_projection["pixel_x"],
                    "center_pixel_y": center_projection["pixel_y"],
                    "crack_world_x_m": float(center[0]),
                }
            )
            best = choose_better(best, candidate)

    status = "not_visible"
    if best is not None:
        status = "full_bbox" if best["full_bbox_visible"] else "center_only"

    result = {
        "crack_id": row["crack_id"],
        "severity": row.get("severity", ""),
        "target_diagonal_mm": row.get("target_diagonal_mm", ""),
        "target_world_x_m": row.get("target_world_x_m", ""),
        "target_theta_deg": row.get("target_theta_deg", ""),
        "status": status,
        "best_camera": "",
        "best_robot_x_m": "",
        "best_center_depth_m": "",
        "visible_corner_count": "0",
        "pixel_bbox_width_px": "",
        "pixel_bbox_height_px": "",
        "pixel_bbox_diag_px": "",
        "center_pixel_x": "",
        "center_pixel_y": "",
        "min_margin_px": "",
    }
    if best is not None:
        result.update(
            {
                "best_camera": best["camera"],
                "best_robot_x_m": f"{best['robot_x_m']:.3f}",
                "best_center_depth_m": f"{best['center_depth_m']:.3f}",
                "visible_corner_count": str(best["visible_corner_count"]),
                "pixel_bbox_width_px": f"{best['bbox_w_px']:.3f}",
                "pixel_bbox_height_px": f"{best['bbox_h_px']:.3f}",
                "pixel_bbox_diag_px": f"{best['bbox_diag_px']:.3f}",
                "center_pixel_x": f"{best['center_pixel_x']:.3f}",
                "center_pixel_y": f"{best['center_pixel_y']:.3f}",
                "min_margin_px": f"{best['margin_px']:.3f}",
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Return a non-zero exit code unless every crack bbox is visible.",
    )
    parser.add_argument(
        "--require-center",
        action="store_true",
        help="Return a non-zero exit code unless every crack center is visible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Coverage input CSV not found: {args.input}")

    cameras = build_camera_models()
    robot_xs = np.arange(
        TUNNEL_X_MIN_M,
        TUNNEL_X_MAX_M + ROBOT_X_SAMPLE_STEP_M * 0.5,
        ROBOT_X_SAMPLE_STEP_M
    )

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = [
        evaluate_crack(row, cameras, robot_xs)
        for row in rows
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    full_count = sum(row["status"] == "full_bbox" for row in results)
    center_count = sum(row["status"] != "not_visible" for row in results)
    print(
        "Camera coverage: "
        f"{full_count}/{len(results)} full bbox, "
        f"{center_count}/{len(results)} center visible"
    )
    print(f"Saved coverage report: {args.output}")

    for row in results:
        print(
            f"{row['crack_id']}: {row['status']} "
            f"camera={row['best_camera']} "
            f"robot_x={row['best_robot_x_m']} "
            f"bbox_diag_px={row['pixel_bbox_diag_px']}"
        )

    if args.require_full and full_count != len(results):
        raise SystemExit(1)
    if args.require_center and center_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

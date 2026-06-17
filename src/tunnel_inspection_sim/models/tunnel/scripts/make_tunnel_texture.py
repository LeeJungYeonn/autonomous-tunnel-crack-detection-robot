import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# =========================
# 경로 설정
# =========================
MODEL_DIR = Path(__file__).resolve().parents[1]

GT_JSON = MODEL_DIR / "gt/cracks_uv_gt.json"

# 기존 터널 texture가 있으면 여기에 둔다.
# 없으면 자동으로 회색 texture 생성.
BASE_TEXTURE = MODEL_DIR / "textures/tunnel_base.png"

OUTPUT_TEXTURE = MODEL_DIR / "tunnel_with_cracks.png"
OUTPUT_SIZE_GT = MODEL_DIR / "cracks/crack_texture_size_gt.csv"

# base texture가 없을 때 생성할 texture 크기
TEXTURE_W = 4096
TEXTURE_H = 4096

# 기본 터널 색상 RGB
BASE_COLOR = (200, 200, 200)

# 실제 터널 모델 스케일:
#   x: -5~5 m = 길이 10 m
#   inner radius: 약 0.70 m = 내경 1.4 m
TUNNEL_LENGTH_M = 10.0
TUNNEL_INNER_RADIUS_M = 0.70
TUNNEL_INNER_DIAMETER_MM = TUNNEL_INNER_RADIUS_M * 2.0 * 1000.0
TUNNEL_UNROLLED_ARC_MM = math.pi * TUNNEL_INNER_RADIUS_M * 1000.0

# 터널 반원통 mesh의 UV는 PNG 왼쪽 절반(u=0.0~0.5)만 사용한다.
# u: 옆면~천장~반대쪽 옆면 방향, v: 입구~출구 방향
TEXTURE_U_MIN = 0.0
TEXTURE_U_MAX = 0.5
TEXTURE_V_MIN = 0.0
TEXTURE_V_MAX = 1.0

# 위험도 기준
# safe: < 80 mm, caution: 80 mm 이상 160 mm 미만, danger: 160 mm 이상
SAFE_MAX_DIAGONAL_MM = 80.0
DANGER_MIN_DIAGONAL_MM = 160.0

# 기준 주변/내부/외부가 위치 순서와 무관하게 섞이도록 배치한다.
CRACK_TARGET_DIAGONAL_MM = {
    "crack_001": 115.0,  # caution
    "crack_002": 230.0,  # danger
    "crack_003": 65.0,   # safe
    "crack_004": 155.0,  # caution, threshold 근처
    "crack_005": 45.0,   # safe
    "crack_006": 180.0,  # danger
    "crack_007": 90.0,   # caution
    "crack_008": 75.0,   # safe, threshold 근처
    "crack_009": 300.0,  # danger
    "crack_010": 140.0,  # caution
}
DEFAULT_TARGET_DIAGONAL_MM = 115.0

# U는 PNG 왼쪽 절반 안의 반원통 전개 방향, V는 터널 길이 방향이다.
# 원래 GT 중심이 한 구간에 몰려 있으므로, 합성 직전에 중심을 터널 전체로 재배치한다.
# 크기와 위치가 정렬돼 보이지 않도록 안전/주의/위험 샘플을 섞어서 둔다.
CRACK_TARGET_UV_CENTER = {
    "crack_001": (0.12, 0.14),
    "crack_002": (0.42, 0.82),
    "crack_003": (0.31, 0.36),
    "crack_004": (0.20, 0.68),
    "crack_005": (0.45, 0.50),
    "crack_006": (0.10, 0.88),
    "crack_007": (0.38, 0.22),
    "crack_008": (0.24, 0.08),
    "crack_009": (0.32, 0.94),
    "crack_010": (0.08, 0.58),
}

# 작은 균열은 padding이 크기 오차를 크게 만들기 때문에 검증용 texture에서는 작게 둔다.
COMPOSITE_PADDING_PX = 1
ALPHA_CROP_MARGIN_PX = 2
# =========================


def uv_to_pixel(uv, width, height):
    """
    Blender/Gazebo UV:
      u: left -> right
      v: bottom -> top

    Image pixel:
      x: left -> right
      y: top -> bottom

    그래서 y는 1-v로 뒤집어야 함.
    """
    u, v = uv
    x = u * (width - 1)
    y = (1.0 - v) * (height - 1)
    return [float(x), float(y)]


def load_base_texture():
    if BASE_TEXTURE.exists():
        img = Image.open(BASE_TEXTURE).convert("RGBA")
        print(f"Loaded base texture: {BASE_TEXTURE}")
        return np.array(img)

    print("Base texture not found. Creating plain gray texture.")
    base = np.zeros((TEXTURE_H, TEXTURE_W, 4), dtype=np.uint8)
    base[:, :, 0] = BASE_COLOR[0]
    base[:, :, 1] = BASE_COLOR[1]
    base[:, :, 2] = BASE_COLOR[2]
    base[:, :, 3] = 255
    return base


def resolve_source_image_path(source_image_path, crack_id):
    """
    GT JSON에 다른 PC의 absolute path가 남아 있어도 현재 model/cracks 아래 파일을 찾는다.
    """
    candidates = []

    if source_image_path:
        src_path = Path(source_image_path)
        candidates.append(src_path)
        candidates.append(MODEL_DIR / "cracks" / src_path.name)

    candidates.append(MODEL_DIR / "cracks" / f"{crack_id}.png")

    for path in candidates:
        if path.exists():
            return path

    return candidates[0] if candidates else None


def crop_to_alpha_bbox(crack_rgba, margin_px=2):
    """
    Transparent margin을 제거해서 target bbox가 실제 보이는 균열 크기에 가깝게 대응하도록 한다.
    """
    alpha = crack_rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        return crack_rgba

    h, w = alpha.shape
    x_min = max(0, int(xs.min()) - margin_px)
    x_max = min(w, int(xs.max()) + 1 + margin_px)
    y_min = max(0, int(ys.min()) - margin_px)
    y_max = min(h, int(ys.max()) + 1 + margin_px)

    return crack_rgba[y_min:y_max, x_min:x_max, :]


def alpha_composite_rgba(base_rgba, overlay_rgba):
    """
    base_rgba 위에 overlay_rgba를 alpha blending.
    둘 다 uint8 RGBA.
    """
    base = base_rgba.astype(np.float32) / 255.0
    overlay = overlay_rgba.astype(np.float32) / 255.0

    alpha = overlay[:, :, 3:4]
    out_rgb = overlay[:, :, :3] * alpha + base[:, :, :3] * (1.0 - alpha)
    out_a = np.ones_like(base[:, :, 3:4])

    out = np.concatenate([out_rgb, out_a], axis=2)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def warp_crack_to_texture(crack_img_rgba, src_uvs, dst_uvs, out_w, out_h):
    """
    crack image를 source UV -> tunnel target UV로 perspective warp.
    """
    h, w = crack_img_rgba.shape[:2]

    src_pts = np.array(
        [uv_to_pixel(uv, w, h) for uv in src_uvs],
        dtype=np.float32
    )

    dst_pts = np.array(
        [uv_to_pixel(uv, out_w, out_h) for uv in dst_uvs],
        dtype=np.float32
    )

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        crack_img_rgba,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return warped

def sort_pairs_by_source_uv(pairs):
    """
    source_uv 기준으로 4개 점을 일관된 순서로 정렬.
    반환 순서:
      bottom-left, bottom-right, top-right, top-left

    source_uv:
      u: left -> right
      v: bottom -> top
    """
    pts = pairs[:4]

    # v 기준으로 아래쪽 2개, 위쪽 2개 분리
    pts_sorted = sorted(pts, key=lambda p: p["source_uv"][1])

    bottom = sorted(pts_sorted[:2], key=lambda p: p["source_uv"][0])
    top = sorted(pts_sorted[2:], key=lambda p: p["source_uv"][0])

    bottom_left = bottom[0]
    bottom_right = bottom[1]
    top_left = top[0]
    top_right = top[1]

    return [bottom_left, bottom_right, top_right, top_left]


def measure_ordered_quad_m(ordered_pairs):
    """
    정렬된 4개 corner의 현재 3D 크기를 계산한다.
    반환: width, height, diagonal (meter)
    """
    pts = np.array([p["world_point"] for p in ordered_pairs], dtype=np.float64)
    bottom_left, bottom_right, top_right, top_left = pts

    width_m = 0.5 * (
        np.linalg.norm(bottom_right - bottom_left)
        + np.linalg.norm(top_right - top_left)
    )
    height_m = 0.5 * (
        np.linalg.norm(top_left - bottom_left)
        + np.linalg.norm(top_right - bottom_right)
    )
    diagonal_m = max(
        np.linalg.norm(top_right - bottom_left),
        np.linalg.norm(top_left - bottom_right),
    )

    return width_m, height_m, diagonal_m


def tunnel_point_from_uv(uv):
    u, v = uv
    theta = (
        (u - TEXTURE_U_MIN)
        / (TEXTURE_U_MAX - TEXTURE_U_MIN)
        * math.pi
    )
    x = v * TUNNEL_LENGTH_M - TUNNEL_LENGTH_M / 2.0
    y = TUNNEL_INNER_RADIUS_M * math.cos(theta)
    z = TUNNEL_INNER_RADIUS_M * math.sin(theta)
    return np.array([x, y, z], dtype=np.float64)


def measure_ordered_uv_quad_m(dst_uvs):
    pts = np.array([tunnel_point_from_uv(uv) for uv in dst_uvs])
    bottom_left, bottom_right, top_right, top_left = pts

    width_m = 0.5 * (
        np.linalg.norm(bottom_right - bottom_left)
        + np.linalg.norm(top_right - top_left)
    )
    height_m = 0.5 * (
        np.linalg.norm(top_left - bottom_left)
        + np.linalg.norm(top_right - bottom_right)
    )
    diagonal_m = max(
        np.linalg.norm(top_right - bottom_left),
        np.linalg.norm(top_left - bottom_right),
    )

    return width_m, height_m, diagonal_m


def scale_uvs_about_center(dst_uvs, scale):
    dst = np.array(dst_uvs, dtype=np.float64)
    center = dst.mean(axis=0)
    scaled = center + (dst - center) * scale
    return scaled.tolist()


def scale_uvs_to_target_diagonal(dst_uvs, target_diag_mm):
    _, _, current_diag_m = measure_ordered_uv_quad_m(dst_uvs)
    if current_diag_m <= 0:
        return dst_uvs, 0.0, 0.0

    scale = (target_diag_mm / 1000.0) / current_diag_m
    scaled = scale_uvs_about_center(dst_uvs, scale)

    for _ in range(4):
        _, _, scaled_diag_m = measure_ordered_uv_quad_m(scaled)
        if scaled_diag_m <= 0:
            break
        correction = (target_diag_mm / 1000.0) / scaled_diag_m
        scale *= correction
        scaled = scale_uvs_about_center(dst_uvs, scale)

    _, _, final_diag_m = measure_ordered_uv_quad_m(scaled)
    return scaled, scale, final_diag_m


def move_uvs_to_center(dst_uvs, target_center):
    if target_center is None:
        return dst_uvs

    dst = np.array(dst_uvs, dtype=np.float64)
    target = np.array(target_center, dtype=np.float64)
    current = dst.mean(axis=0)
    dst += target - current
    return dst.tolist()


def classify_crack_by_diagonal(diagonal_mm):
    if diagonal_mm < SAFE_MAX_DIAGONAL_MM:
        return "safe", "green"
    if diagonal_mm < DANGER_MIN_DIAGONAL_MM:
        return "caution", "orange"
    return "danger", "red"


def uv_bbox(dst_uvs):
    us = [uv[0] for uv in dst_uvs]
    vs = [uv[1] for uv in dst_uvs]
    return min(us), max(us), min(vs), max(vs)


def is_reasonable_target_uv(dst_uvs, max_uv_size=0.25):
    """
    target_uv가 비정상적으로 큰 영역으로 잡힌 경우 skip하기 위한 검사.
    """
    us = [uv[0] for uv in dst_uvs]
    vs = [uv[1] for uv in dst_uvs]

    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    width = u_max - u_min
    height = v_max - v_min

    if width <= 0 or height <= 0:
        return False

    if width > max_uv_size or height > max_uv_size:
        return False

    if u_min < TEXTURE_U_MIN or u_max > TEXTURE_U_MAX:
        return False

    if v_min < TEXTURE_V_MIN or v_max > TEXTURE_V_MAX:
        return False

    return True

def composite_crack_by_uv_bbox(canvas, crack_rgba, dst_uvs, padding_px=2):
    """
    Perspective transform을 쓰지 않고,
    target UV의 bounding box 영역에 crack PNG를 resize해서 alpha composite.
    점 순서 꼬임으로 생기는 긴 대각선 artifact를 방지하는 안전한 방식.
    """
    out_h, out_w = canvas.shape[:2]

    # target UV -> pixel
    dst_pixels = np.array(
        [uv_to_pixel(uv, out_w, out_h) for uv in dst_uvs],
        dtype=np.float32
    )

    x_min = int(np.floor(dst_pixels[:, 0].min())) - padding_px
    x_max = int(np.ceil(dst_pixels[:, 0].max())) + padding_px
    y_min = int(np.floor(dst_pixels[:, 1].min())) - padding_px
    y_max = int(np.ceil(dst_pixels[:, 1].max())) + padding_px

    # image boundary clamp
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(out_w - 1, x_max)
    y_max = min(out_h - 1, y_max)

    paste_w = x_max - x_min + 1
    paste_h = y_max - y_min + 1

    if paste_w <= 1 or paste_h <= 1:
        return canvas, False

    resized = cv2.resize(
        crack_rgba,
        (paste_w, paste_h),
        interpolation=cv2.INTER_AREA
    )

    roi = canvas[y_min:y_max + 1, x_min:x_max + 1, :]

    blended = alpha_composite_rgba(roi, resized)
    canvas[y_min:y_max + 1, x_min:x_max + 1, :] = blended

    return canvas, True


def main():
    if not GT_JSON.exists():
        raise FileNotFoundError(f"GT JSON not found: {GT_JSON}")

    OUTPUT_TEXTURE.parent.mkdir(parents=True, exist_ok=True)

    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt = json.load(f)

    canvas = load_base_texture()
    out_h, out_w = canvas.shape[:2]

    cracks = gt["cracks"]
    print(f"Number of cracks: {len(cracks)}")
    print(
        "Severity thresholds: "
        f"safe < {SAFE_MAX_DIAGONAL_MM:.0f} mm, "
        f"caution < {DANGER_MIN_DIAGONAL_MM:.0f} mm, "
        f"danger >= {DANGER_MIN_DIAGONAL_MM:.0f} mm"
    )
    print(
        "Tunnel scale: "
        f"length={TUNNEL_LENGTH_M:.2f} m, "
        f"inner_diameter={TUNNEL_INNER_DIAMETER_MM:.0f} mm, "
        f"unrolled_arc={TUNNEL_UNROLLED_ARC_MM:.0f} mm, "
        f"texture_u={TEXTURE_U_MIN:.1f}~{TEXTURE_U_MAX:.1f}"
    )

    size_rows = []

    for crack in cracks:
        crack_id = crack["crack_id"]
        source_image_path = crack["source_image_path"]

        if source_image_path is None:
            print(f"[SKIP] {crack_id}: source_image_path is None")
            continue

        src_path = resolve_source_image_path(source_image_path, crack_id)

        if src_path is None or not src_path.exists():
            print(f"[SKIP] {crack_id}: source image not found: {src_path}")
            continue

        crack_img = Image.open(src_path).convert("RGBA")
        crack_rgba = crop_to_alpha_bbox(np.array(crack_img), margin_px=ALPHA_CROP_MARGIN_PX)

        pairs = crack["uv_pairs"]

        if len(pairs) < 4:
            print(f"[SKIP] {crack_id}: fewer than 4 uv pairs")
            continue

        ordered_pairs = sort_pairs_by_source_uv(pairs)
        dst_uvs = [p["target_uv"] for p in ordered_pairs]
        current_width_m, current_height_m, current_diag_m = (
            measure_ordered_uv_quad_m(dst_uvs)
        )

        if current_diag_m <= 0:
            print(f"[SKIP] {crack_id}: invalid current diagonal")
            continue

        target_diag_mm = CRACK_TARGET_DIAGONAL_MM.get(
            crack_id,
            DEFAULT_TARGET_DIAGONAL_MM
        )
        dst_uvs, scale, _ = scale_uvs_to_target_diagonal(
            dst_uvs,
            target_diag_mm
        )
        target_uv_center = CRACK_TARGET_UV_CENTER.get(crack_id)
        dst_uvs = move_uvs_to_center(dst_uvs, target_uv_center)
        target_width_m, target_height_m, target_diag_m = (
            measure_ordered_uv_quad_m(dst_uvs)
        )
        target_width_mm = target_width_m * 1000.0
        target_height_mm = target_height_m * 1000.0
        severity, marker_color = classify_crack_by_diagonal(target_diag_mm)

        us = [p[0] for p in dst_uvs]
        vs = [p[1] for p in dst_uvs]
        u_center = float(np.mean(us))
        v_center = float(np.mean(vs))
        world_x_m = v_center * TUNNEL_LENGTH_M - TUNNEL_LENGTH_M / 2.0
        theta_deg = (
            (u_center - TEXTURE_U_MIN)
            / (TEXTURE_U_MAX - TEXTURE_U_MIN)
            * 180.0
        )
        print(
            f"[DEBUG] {crack_id}: "
            f"u_range={min(us):.4f}~{max(us):.4f}, "
            f"v_range={min(vs):.4f}~{max(vs):.4f}, "
            f"size=({max(us)-min(us):.4f}, {max(vs)-min(vs):.4f}), "
            f"center=({u_center:.2f}, {v_center:.2f}), "
            f"world_x={world_x_m:.2f}m, theta={theta_deg:.1f}deg, "
            f"target_diag={target_diag_m * 1000.0:.1f}mm, "
            f"class={severity}"
        )

        if not is_reasonable_target_uv(dst_uvs, max_uv_size=0.25):
            print(f"[SKIP] {crack_id}: unreasonable target UV")
            continue

        canvas, ok = composite_crack_by_uv_bbox(
            canvas,
            crack_rgba,
            dst_uvs,
            padding_px=COMPOSITE_PADDING_PX
        )

        if not ok:
            print(f"[SKIP] {crack_id}: bbox too small")
            continue

        print(f"[OK] composited {crack_id} by bbox")
        u_min, u_max, v_min, v_max = uv_bbox(dst_uvs)
        size_rows.append({
            "crack_id": crack_id,
            "image_file": src_path.name,
            "severity": severity,
            "marker_color": marker_color,
            "safe_max_diagonal_mm": f"{SAFE_MAX_DIAGONAL_MM:.3f}",
            "danger_min_diagonal_mm": f"{DANGER_MIN_DIAGONAL_MM:.3f}",
            "target_diagonal_mm": f"{target_diag_m * 1000.0:.3f}",
            "target_width_mm": f"{target_width_mm:.3f}",
            "target_height_mm": f"{target_height_mm:.3f}",
            "target_diagonal_to_tunnel_diameter": (
                f"{target_diag_mm / TUNNEL_INNER_DIAMETER_MM:.6f}"
            ),
            "target_u_center": f"{u_center:.6f}",
            "target_v_center": f"{v_center:.6f}",
            "target_world_x_m": f"{world_x_m:.3f}",
            "target_theta_deg": f"{theta_deg:.3f}",
            "original_diagonal_mm": f"{current_diag_m * 1000.0:.3f}",
            "scale_factor": f"{scale:.6f}",
            "u_min": f"{u_min:.9f}",
            "u_max": f"{u_max:.9f}",
            "v_min": f"{v_min:.9f}",
            "v_max": f"{v_max:.9f}",
        })

    # Gazebo OBJ/MTL에서 쓰기 좋게 RGB로 저장
    out_img = Image.fromarray(canvas).convert("RGB")
    out_img.save(OUTPUT_TEXTURE)

    print(f"Saved output texture: {OUTPUT_TEXTURE}")

    if size_rows:
        with open(OUTPUT_SIZE_GT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(size_rows[0].keys()))
            writer.writeheader()
            writer.writerows(size_rows)
        print(f"Saved crack size GT: {OUTPUT_SIZE_GT}")


if __name__ == "__main__":
    main()

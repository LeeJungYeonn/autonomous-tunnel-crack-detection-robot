import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# =========================
# 경로 설정
# =========================
MODEL_DIR = Path.home() / "tunnel_ws/src/tunnel_inspection_sim/models/tunnel"

GT_JSON = MODEL_DIR / "gt/cracks_uv_gt.json"

# 기존 터널 texture가 있으면 여기에 둔다.
# 없으면 자동으로 회색 texture 생성.
BASE_TEXTURE = MODEL_DIR / "textures/tunnel_base.png"

OUTPUT_TEXTURE = MODEL_DIR / "tunnel_with_cracks.png"

# base texture가 없을 때 생성할 texture 크기
TEXTURE_W = 4096
TEXTURE_H = 4096

# 기본 터널 색상 RGB
BASE_COLOR = (200, 200, 200)
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

    # UV가 0~1 범위에서 너무 벗어나면 skip
    if u_min < -0.05 or u_max > 1.05:
        return False

    if v_min < -0.05 or v_max > 1.05:
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

    for crack in cracks:
        crack_id = crack["crack_id"]
        source_image_path = crack["source_image_path"]

        if source_image_path is None:
            print(f"[SKIP] {crack_id}: source_image_path is None")
            continue

        src_path = Path(source_image_path)

        if not src_path.exists():
            print(f"[SKIP] {crack_id}: source image not found: {src_path}")
            continue

        crack_img = Image.open(src_path).convert("RGBA")
        crack_rgba = np.array(crack_img)

        pairs = crack["uv_pairs"]

        if len(pairs) < 4:
            print(f"[SKIP] {crack_id}: fewer than 4 uv pairs")
            continue

        ordered_pairs = sort_pairs_by_source_uv(pairs)
        dst_uvs = [p["target_uv"] for p in ordered_pairs]

        us = [p[0] for p in dst_uvs]
        vs = [p[1] for p in dst_uvs]
        print(
            f"[DEBUG] {crack_id}: "
            f"u_range={min(us):.4f}~{max(us):.4f}, "
            f"v_range={min(vs):.4f}~{max(vs):.4f}, "
            f"size=({max(us)-min(us):.4f}, {max(vs)-min(vs):.4f})"
        )

        if not is_reasonable_target_uv(dst_uvs, max_uv_size=0.25):
            print(f"[SKIP] {crack_id}: unreasonable target UV")
            continue

        canvas, ok = composite_crack_by_uv_bbox(
            canvas,
            crack_rgba,
            dst_uvs,
            padding_px=40
        )

        if not ok:
            print(f"[SKIP] {crack_id}: bbox too small")
            continue

        print(f"[OK] composited {crack_id} by bbox")

    # Gazebo OBJ/MTL에서 쓰기 좋게 RGB로 저장
    out_img = Image.fromarray(canvas).convert("RGB")
    out_img.save(OUTPUT_TEXTURE)

    print(f"Saved output texture: {OUTPUT_TEXTURE}")


if __name__ == "__main__":
    main()
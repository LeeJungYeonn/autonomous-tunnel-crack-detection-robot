import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# TF2 관련 패키지
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point


SAFE_MAX_DIAGONAL_MM = 80.0
DANGER_MIN_DIAGONAL_MM = 160.0
SEVERITY_COLORS_BGR = {
    'safe': (0, 180, 0),
    'caution': (0, 165, 255),
    'danger': (0, 0, 255),
}


class CrackDetectorNode(Node):
    def __init__(self):
        super().__init__('crack_detector_node')
        self.bridge = CvBridge()

        # [신규 모델 적용] Hugging Face에서 다운받은 모델 경로
        model_path = os.path.expanduser(
            '~/tunnel_ws/src/tunnel_inspection_sim/models/yolov8_crack_seg.pt'
        )
        self.get_logger().info(f"YOLO 모델 로딩: {model_path}")
        self.yolo_model = YOLO(model_path)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 양쪽 RGB-D 카메라를 각각 동기화해서 같은 탐지/매핑 파이프라인으로 처리한다.
        self.camera_subscribers = []
        self.camera_synchronizers = []
        self.logged_camera_frames = set()
        self.mapped_log_counts = {}
        self.detection_log_counts = {}
        self.tf_fallback_log_counts = {}
        self.camera_names = ('left', 'right', 'top')
        for camera_name in self.camera_names:
            self.create_camera_synchronizer(camera_name)

        # [시각화 캔버스 설정]
        self.map_w, self.map_h = 1100, 430
        self.map_margin_left = 55
        self.map_margin_right = 25
        self.map_margin_top = 36
        self.map_margin_bottom = 115
        self.map_plot_w = (
            self.map_w - self.map_margin_left - self.map_margin_right
        )
        self.map_plot_h = (
            self.map_h - self.map_margin_top - self.map_margin_bottom
        )
        self.tunnel_x_min = float(
            self.declare_parameter('tunnel_x_min', -5.0).value
        )
        self.tunnel_x_max = float(
            self.declare_parameter('tunnel_x_max', 5.0).value
        )
        self.odom_origin_world_x = float(
            self.declare_parameter('odom_origin_world_x', -5.0).value
        )
        self.odom_origin_world_y = float(
            self.declare_parameter('odom_origin_world_y', 0.0).value
        )
        self.odom_origin_world_z = float(
            self.declare_parameter('odom_origin_world_z', 0.09).value
        )
        self.default_conf = float(
            self.declare_parameter('default_conf', 0.2).value
        )
        self.side_conf = float(
            self.declare_parameter('side_conf', 0.08).value
        )
        self.top_conf = float(
            self.declare_parameter('top_conf', 0.05).value
        )
        self.inference_imgsz = int(
            self.declare_parameter('inference_imgsz', 960).value
        )
        self.use_latest_tf = bool(
            self.declare_parameter('use_latest_tf', True).value
        )
        self.max_detection_depth_m = float(
            self.declare_parameter('max_detection_depth_m', 1.2).value
        )
        self.edge_reject_px = int(
            self.declare_parameter('edge_reject_px', 4).value
        )
        self.max_bbox_area_ratio = float(
            self.declare_parameter('max_bbox_area_ratio', 0.35).value
        )
        self.map_cluster_merge_px = int(
            self.declare_parameter('map_cluster_merge_px', 55).value
        )
        self.safe_max_diagonal_mm = float(
            self.declare_parameter(
                'safe_max_diagonal_mm',
                SAFE_MAX_DIAGONAL_MM
            ).value
        )
        self.danger_min_diagonal_mm = float(
            self.declare_parameter(
                'danger_min_diagonal_mm',
                DANGER_MIN_DIAGONAL_MM
            ).value
        )
        self.tunnel_length = self.tunnel_x_max - self.tunnel_x_min
        if self.tunnel_length <= 0.0:
            self.get_logger().warn("터널 x 범위가 잘못되어 기본값(-5.0~5.0)을 사용합니다.")
            self.tunnel_x_min = -5.0
            self.tunnel_x_max = 5.0
            self.tunnel_length = 10.0

        self.map_crack_clusters = []
        self.unrolled_map = self.create_unrolled_map()
        self.get_logger().info(
            "시스템 준비 완료! "
            f"(터널 X: {self.tunnel_x_min:.1f}~{self.tunnel_x_max:.1f}, "
            f"odom 원점 월드: "
            f"({self.odom_origin_world_x:.1f}, "
            f"{self.odom_origin_world_y:.1f}, "
            f"{self.odom_origin_world_z:.2f}), "
            f"카메라: {'/'.join(self.camera_names)}, "
            f"conf: default={self.default_conf:.2f}, "
            f"side={self.side_conf:.2f}, top={self.top_conf:.2f}, "
            f"imgsz={self.inference_imgsz}, "
            f"TF={'latest' if self.use_latest_tf else 'timestamp'}, "
            f"max_depth={self.max_detection_depth_m:.2f}m, "
            f"위험도 기준: safe<{self.safe_max_diagonal_mm:.0f}mm, "
            f"danger>={self.danger_min_diagonal_mm:.0f}mm)"
        )

    def create_unrolled_map(self):
        canvas = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 255
        plot_left = self.map_margin_left
        plot_top = self.map_margin_top
        plot_right = self.map_margin_left + self.map_plot_w - 1
        plot_bottom = self.map_margin_top + self.map_plot_h - 1

        cv2.rectangle(
            canvas,
            (plot_left, plot_top),
            (plot_right, plot_bottom),
            (246, 246, 246),
            -1
        )
        cv2.rectangle(
            canvas,
            (plot_left, plot_top),
            (plot_right, plot_bottom),
            (80, 80, 80),
            1
        )

        cv2.putText(
            canvas,
            "Tunnel Unrolled Map",
            (plot_left, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            2
        )
        cv2.putText(
            canvas,
            "x position from tunnel start",
            (plot_left + 250, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (90, 90, 90),
            1
        )

        tick_count = int(math.floor(self.tunnel_length))
        for meter in range(tick_count + 1):
            u = meter / self.tunnel_length
            px, _ = self.map_uv_to_pixel(u, 0.0)
            grid_color = (210, 210, 210)
            if meter == 0 or meter == tick_count:
                grid_color = (150, 150, 150)
            cv2.line(
                canvas,
                (px, plot_top),
                (px, plot_bottom),
                grid_color,
                1
            )
            cv2.line(
                canvas,
                (px, plot_bottom),
                (px, plot_bottom + 6),
                (80, 80, 80),
                1
            )
            label = f"{meter}m"
            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                1
            )
            cv2.putText(
                canvas,
                label,
                (px - text_size[0] // 2, plot_bottom + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (50, 50, 50),
                1
            )

        cv2.putText(
            canvas,
            "0m = robot spawn / tunnel start",
            (plot_left, self.map_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1
        )
        self.draw_severity_legend(canvas)
        return canvas

    def draw_severity_legend(self, canvas):
        entries = [
            ('safe <80mm', SEVERITY_COLORS_BGR['safe']),
            ('caution 80-160mm', SEVERITY_COLORS_BGR['caution']),
            ('danger >=160mm', SEVERITY_COLORS_BGR['danger']),
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1
        total_w = 0
        for label, _ in entries:
            text_size, _ = cv2.getTextSize(label, font, scale, thickness)
            total_w += 18 + text_size[0]
        total_w += 24 * (len(entries) - 1)

        x = self.map_w - total_w - 12
        y = 19
        for label, color in entries:
            cv2.circle(canvas, (x, y - 4), 5, color, -1)
            cv2.putText(
                canvas,
                label,
                (x + 12, y),
                font,
                scale,
                (70, 70, 70),
                thickness
            )
            text_size, _ = cv2.getTextSize(label, font, scale, thickness)
            x += 18 + text_size[0] + 24

    def map_uv_to_pixel(self, u, v):
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        px = self.map_margin_left + int(round(u * (self.map_plot_w - 1)))
        py = self.map_margin_top + int(round(v * (self.map_plot_h - 1)))
        return px, py

    def draw_text_with_background(self, image, text, origin, color):
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size
        x = max(2, min(x, image.shape[1] - text_w - 6))
        y = max(text_h + 4, min(y, image.shape[0] - baseline - 4))

        cv2.rectangle(
            image,
            (x - 3, y - text_h - 3),
            (x + text_w + 3, y + baseline + 3),
            (255, 255, 255),
            -1
        )
        cv2.rectangle(
            image,
            (x - 3, y - text_h - 3),
            (x + text_w + 3, y + baseline + 3),
            (210, 210, 210),
            1
        )
        cv2.putText(image, text, (x, y), font, scale, color, thickness)

    def draw_rotated_text(self, image, text, origin, color, angle=45):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size
        pad = 6
        src_h = text_h + baseline + pad * 2
        src_w = text_w + pad * 2
        text_img = np.ones((src_h, src_w, 3), dtype=np.uint8) * 255
        cv2.putText(
            text_img,
            text,
            (pad, pad + text_h),
            font,
            scale,
            color,
            thickness
        )

        side = int(math.ceil(math.sqrt(src_w ** 2 + src_h ** 2))) + 2
        square = np.ones((side, side, 3), dtype=np.uint8) * 255
        x0 = (side - src_w) // 2
        y0 = (side - src_h) // 2
        square[y0:y0 + src_h, x0:x0 + src_w] = text_img

        center = (side / 2.0, side / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            square,
            matrix,
            (side, side),
            flags=cv2.INTER_LINEAR,
            borderValue=(255, 255, 255)
        )
        mask = np.any(rotated < 245, axis=2)
        if not np.any(mask):
            return

        ys, xs = np.where(mask)
        crop_x1, crop_x2 = xs.min(), xs.max() + 1
        crop_y1, crop_y2 = ys.min(), ys.max() + 1
        crop = rotated[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        dst_x = int(round(origin[0]))
        dst_y = int(round(origin[1]))
        dst_x = max(0, min(dst_x, image.shape[1] - crop.shape[1]))
        dst_y = max(0, min(dst_y, image.shape[0] - crop.shape[0]))
        roi = image[
            dst_y:dst_y + crop.shape[0],
            dst_x:dst_x + crop.shape[1]
        ]
        roi[crop_mask] = crop[crop_mask]

    def draw_vertical_dashed_line(self, image, x, y1, y2, color):
        dash_len = 7
        gap_len = 6
        y = y1
        while y <= y2:
            cv2.line(
                image,
                (x, y),
                (x, min(y + dash_len, y2)),
                color,
                1
            )
            y += dash_len + gap_len

    def draw_crack_axis_marker(self, image, px, distance_m, color, label_slots):
        plot_bottom = self.map_margin_top + self.map_plot_h - 1
        label = f"{distance_m:.1f}m"
        label_y = plot_bottom + 34
        while any(
            abs(px - slot_px) < 42 and abs(label_y - slot_y) < 18
            for slot_px, slot_y in label_slots
        ):
            label_y += 18
        label_slots.append((px, label_y))

        cv2.line(
            image,
            (px, plot_bottom),
            (px, plot_bottom + 9),
            color,
            1
        )
        self.draw_rotated_text(
            image,
            label,
            (px - 8, label_y),
            color,
            angle=45
        )

    def find_map_crack_cluster(self, px, py):
        best_cluster = None
        best_dist_sq = self.map_cluster_merge_px * self.map_cluster_merge_px
        for cluster in self.map_crack_clusters:
            center_px = float(np.mean(cluster['px']))
            center_py = float(np.mean(cluster['py']))
            dx = px - center_px
            dy = py - center_py
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_cluster = cluster
        return best_cluster

    def add_map_detection(
        self,
        px,
        py,
        distance_from_start_m,
        diagonal_mm,
        length_mm,
        width_mm
    ):
        cluster = self.find_map_crack_cluster(px, py)
        if cluster is None:
            cluster = {
                'px': [],
                'py': [],
                'distance_m': [],
                'diagonal_mm': [],
                'length_mm': [],
                'width_mm': [],
            }
            self.map_crack_clusters.append(cluster)

        cluster['px'].append(px)
        cluster['py'].append(py)
        cluster['distance_m'].append(distance_from_start_m)
        cluster['diagonal_mm'].append(diagonal_mm)
        cluster['length_mm'].append(length_mm)
        cluster['width_mm'].append(width_mm)
        self.render_unrolled_map()

    def render_unrolled_map(self):
        canvas = self.create_unrolled_map()
        plot_top = self.map_margin_top
        plot_bottom = self.map_margin_top + self.map_plot_h - 1
        label_slots = []

        for cluster in self.map_crack_clusters:
            avg_diag_mm = float(np.mean(cluster['diagonal_mm']))
            severity = self.classify_crack(avg_diag_mm)
            color = self.severity_color(severity)
            start_m = min(cluster['distance_m'])
            end_m = max(cluster['distance_m'])
            midpoint_m = 0.5 * (start_m + end_m)
            midpoint_u = midpoint_m / self.tunnel_length
            midpoint_px, _ = self.map_uv_to_pixel(midpoint_u, 0.0)
            self.draw_vertical_dashed_line(
                canvas,
                midpoint_px,
                plot_top,
                plot_bottom,
                color
            )
            self.draw_crack_axis_marker(
                canvas,
                midpoint_px,
                midpoint_m,
                color,
                label_slots
            )

        for cluster in self.map_crack_clusters:
            avg_diag_mm = float(np.mean(cluster['diagonal_mm']))
            severity = self.classify_crack(avg_diag_mm)
            color = self.severity_color(severity)
            for px, py in zip(cluster['px'], cluster['py']):
                cv2.circle(canvas, (px, py), 5, color, -1)

            center_px = int(round(float(np.mean(cluster['px']))))
            center_py = int(round(float(np.mean(cluster['py']))))
            self.draw_text_with_background(
                canvas,
                f"{avg_diag_mm:.0f}mm",
                (center_px + 9, center_py - 8),
                color
            )

        self.unrolled_map = canvas

    def create_camera_synchronizer(self, camera_name):
        topic_prefix = f'/{camera_name}_camera'
        img_sub = message_filters.Subscriber(
            self,
            Image,
            f'{topic_prefix}/image'
        )
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            f'{topic_prefix}/depth'
        )
        info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            f'{topic_prefix}/camera_info'
        )
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [img_sub, depth_sub, info_sub],
            10,
            0.1
        )
        synchronizer.registerCallback(
            lambda img_msg, depth_msg, info_msg, name=camera_name: (
                self.sync_callback(
                    name,
                    img_msg,
                    depth_msg,
                    info_msg
                )
            )
        )
        self.camera_subscribers.extend([img_sub, depth_sub, info_sub])
        self.camera_synchronizers.append(synchronizer)

    def lookup_camera_transform(self, camera_name, source_frame, stamp):
        if self.use_latest_tf:
            return self.tf_buffer.lookup_transform(
                'odom',
                source_frame,
                rclpy.time.Time()
            )

        try:
            return self.tf_buffer.lookup_transform(
                'odom',
                source_frame,
                rclpy.time.Time.from_msg(stamp)
            )
        except Exception as timed_error:
            log_count = self.tf_fallback_log_counts.get(camera_name, 0)
            if log_count < 3:
                self.get_logger().warn(
                    f"{camera_name} timestamp TF lookup 실패, "
                    f"latest TF로 fallback: {timed_error}"
                )
                self.tf_fallback_log_counts[camera_name] = log_count + 1
            return self.tf_buffer.lookup_transform(
                'odom',
                source_frame,
                rclpy.time.Time()
            )

    def is_detection_candidate(self, camera_name, box, image_shape):
        height, width = image_shape[:2]
        u1, v1, u2, v2 = map(int, box.xyxy[0])
        bbox_w = max(0, u2 - u1)
        bbox_h = max(0, v2 - v1)
        if bbox_w <= 1 or bbox_h <= 1:
            return False

        area_ratio = (bbox_w * bbox_h) / float(width * height)
        if area_ratio > self.max_bbox_area_ratio:
            return False

        edge = self.edge_reject_px
        touches_edge = (
            u1 <= edge
            or v1 <= edge
            or u2 >= width - 1 - edge
            or v2 >= height - 1 - edge
        )
        if touches_edge:
            return False

        if camera_name in ('left', 'right') and v2 >= height - 1 - edge:
            return False

        return True

    def mask_to_image_size(self, mask, image_shape):
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        height, width = image_shape[:2]
        if mask_np.shape[:2] != (height, width):
            mask_np = cv2.resize(
                mask_np,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
        return mask_np > 0.5

    def classify_crack(self, diagonal_mm):
        if diagonal_mm < self.safe_max_diagonal_mm:
            return 'safe'
        if diagonal_mm < self.danger_min_diagonal_mm:
            return 'caution'
        return 'danger'

    def severity_color(self, severity):
        return SEVERITY_COLORS_BGR.get(severity, (255, 255, 255))

    def depth_pixels_to_camera_points(self, xs, ys, depths, fx, fy, cx, cy):
        depths = depths.astype(np.float32)
        valid = np.isfinite(depths) & (depths > 0.05)
        if np.count_nonzero(valid) < 3:
            return None

        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        depths = depths[valid]

        median_depth = np.median(depths)
        mad = np.median(np.abs(depths - median_depth))
        depth_tol = max(0.03, 3.0 * mad)
        inliers = np.abs(depths - median_depth) <= depth_tol
        if np.count_nonzero(inliers) >= 3:
            xs = xs[inliers]
            ys = ys[inliers]
            depths = depths[inliers]

        camera_x = (xs - cx) * depths / fx
        camera_y = (ys - cy) * depths / fy
        camera_z = depths
        return np.column_stack(
            (camera_x, camera_y, camera_z)
        ).astype(np.float32)

    def sample_points(self, points, max_points=300):
        if len(points) <= max_points:
            return points
        step = int(math.ceil(len(points) / max_points))
        return points[::step][:max_points]

    def points_to_camera_geometry(self, points, source):
        if points is None or len(points) < 3:
            return None

        center = np.median(points, axis=0)
        centered = points - center

        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axes = vh[:2]
            projected = centered @ axes.T
            extents = (
                np.percentile(projected, 95, axis=0)
                - np.percentile(projected, 5, axis=0)
            )
            length_m = float(max(extents))
            width_m = float(min(extents))
        except np.linalg.LinAlgError:
            length_m = 0.0
            width_m = 0.0

        sampled = self.sample_points(points)
        diffs = sampled[:, None, :] - sampled[None, :, :]
        diagonal_m = float(np.sqrt(np.max(np.sum(diffs * diffs, axis=2))))
        diagonal_m = max(diagonal_m, math.sqrt(length_m ** 2 + width_m ** 2))

        return {
            'center': tuple(float(v) for v in center),
            'length_m': length_m,
            'width_m': width_m,
            'diagonal_m': diagonal_m,
            'source': source,
            'point_count': int(len(points)),
        }

    def depth_patch_to_camera_point(self, u, v, cv_depth, fx, fy, cx, cy):
        height, width = cv_depth.shape[:2]
        radius = 4
        u = int(max(0, min(width - 1, round(u))))
        v = int(max(0, min(height - 1, round(v))))

        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        patch = cv_depth[y0:y1, x0:x1]
        grid_y, grid_x = np.indices(patch.shape)
        points = self.depth_pixels_to_camera_points(
            (grid_x + x0).reshape(-1),
            (grid_y + y0).reshape(-1),
            patch.reshape(-1),
            fx,
            fy,
            cx,
            cy
        )
        if points is None:
            return None
        return np.median(points, axis=0)

    def bbox_corner_geometry(self, u1, v1, u2, v2, cv_depth, fx, fy, cx, cy):
        corner_pixels = [
            (u1, v1),
            (u2, v1),
            (u2, v2),
            (u1, v2),
        ]
        corner_points = []
        for u, v in corner_pixels:
            point = self.depth_patch_to_camera_point(
                u,
                v,
                cv_depth,
                fx,
                fy,
                cx,
                cy
            )
            if point is not None:
                corner_points.append(point)

        if len(corner_points) == 4:
            pts = np.asarray(corner_points, dtype=np.float32)
            top_left, top_right, bottom_right, bottom_left = pts
            width_m = 0.5 * (
                np.linalg.norm(top_right - top_left)
                + np.linalg.norm(bottom_right - bottom_left)
            )
            height_m = 0.5 * (
                np.linalg.norm(bottom_left - top_left)
                + np.linalg.norm(bottom_right - top_right)
            )
            diagonal_m = max(
                np.linalg.norm(bottom_right - top_left),
                np.linalg.norm(bottom_left - top_right)
            )
            return {
                'center': tuple(float(v) for v in np.median(pts, axis=0)),
                'length_m': float(max(width_m, height_m)),
                'width_m': float(min(width_m, height_m)),
                'diagonal_m': float(diagonal_m),
                'source': 'bbox_corners',
                'point_count': 4,
            }

        if len(corner_points) >= 3:
            return self.points_to_camera_geometry(
                np.asarray(corner_points, dtype=np.float32),
                'bbox_partial_corners'
            )

        return None

    def center_depth_bbox_geometry(
        self,
        u1,
        v1,
        u2,
        v2,
        cv_depth,
        fx,
        fy,
        cx,
        cy
    ):
        cu, cv = (u1 + u2) // 2, (v1 + v2) // 2
        center = self.depth_patch_to_camera_point(
            cu,
            cv,
            cv_depth,
            fx,
            fy,
            cx,
            cy
        )
        if center is None:
            return None

        depth = float(center[2])
        corner_pixels = np.asarray(
            [
                (u1, v1),
                (u2, v1),
                (u2, v2),
                (u1, v2),
            ],
            dtype=np.float32
        )
        xs = corner_pixels[:, 0]
        ys = corner_pixels[:, 1]
        depths = np.full(xs.shape, depth, dtype=np.float32)
        points = self.depth_pixels_to_camera_points(
            xs,
            ys,
            depths,
            fx,
            fy,
            cx,
            cy
        )
        if points is None:
            return {
                'center': tuple(float(v) for v in center),
                'length_m': 0.0,
                'width_m': 0.0,
                'diagonal_m': 0.0,
                'source': 'center_depth',
                'point_count': 1,
            }

        geometry = self.points_to_camera_geometry(points, 'bbox_center_depth')
        if geometry is not None:
            geometry['center'] = tuple(float(v) for v in center)
        return geometry

    def detection_to_camera_geometry(
        self,
        box,
        mask,
        cv_depth,
        fx,
        fy,
        cx,
        cy
    ):
        u1, v1, u2, v2 = map(int, box.xyxy[0])

        if mask is not None:
            mask_img = self.mask_to_image_size(mask, cv_depth.shape)
            ys, xs = np.where(mask_img)
            if len(xs) > 0:
                points = self.depth_pixels_to_camera_points(
                    xs,
                    ys,
                    cv_depth[ys, xs],
                    fx,
                    fy,
                    cx,
                    cy
                )
                geometry = self.points_to_camera_geometry(points, 'mask')
                if geometry is not None:
                    return geometry

        geometry = self.bbox_corner_geometry(
            u1,
            v1,
            u2,
            v2,
            cv_depth,
            fx,
            fy,
            cx,
            cy
        )
        if geometry is not None:
            return geometry

        return self.center_depth_bbox_geometry(
            u1,
            v1,
            u2,
            v2,
            cv_depth,
            fx,
            fy,
            cx,
            cy
        )

    def sync_callback(self, camera_name, img_msg, depth_msg, info_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding='32FC1'
        )
        source_frame = f'{camera_name}_camera_optical_frame'
        reported_frame = info_msg.header.frame_id
        if camera_name not in self.logged_camera_frames:
            if reported_frame and reported_frame != source_frame:
                self.get_logger().warn(
                    f"{camera_name} camera_info frame_id는 "
                    f"'{reported_frame}'이지만 "
                    f"depth 좌표는 '{source_frame}'로 변환합니다."
                )
            else:
                self.get_logger().info(
                    f"{camera_name} camera depth 좌표 frame: {source_frame}"
                )
            self.logged_camera_frames.add(camera_name)

        fx, fy, cx, cy = (
            info_msg.k[0],
            info_msg.k[4],
            info_msg.k[2],
            info_msg.k[5]
        )

        # 모델 추론
        conf = self.top_conf if camera_name == 'top' else self.side_conf
        results = self.yolo_model(
            cv_img,
            conf=conf,
            imgsz=self.inference_imgsz,
            verbose=False
        )
        raw_detection_count = sum(len(r.boxes) for r in results)
        detection_log_count = self.detection_log_counts.get(camera_name, 0)
        if detection_log_count < 10:
            self.get_logger().info(
                f"{camera_name} YOLO detections: {raw_detection_count} "
                f"(conf={conf:.2f})"
            )
            self.detection_log_counts[camera_name] = detection_log_count + 1

        for r in results:
            # Segmentation 모델이어도 BBox(네모 박스)는 동일하게 추출 가능!
            masks = r.masks.data if r.masks is not None else []
            for detection_idx, box in enumerate(r.boxes):
                u1, v1, u2, v2 = map(int, box.xyxy[0])
                if not self.is_detection_candidate(
                    camera_name,
                    box,
                    cv_img.shape
                ):
                    continue

                if detection_idx < len(masks):
                    mask = masks[detection_idx]
                else:
                    mask = None

                # 1. 탐지 즉시 파란색 박스 그리기
                cv2.rectangle(cv_img, (u1, v1), (u2, v2), (255, 0, 0), 2)

                camera_geometry = self.detection_to_camera_geometry(
                    box,
                    mask,
                    cv_depth,
                    fx,
                    fy,
                    cx,
                    cy
                )
                if camera_geometry is None:
                    continue
                wx, wy, wz = camera_geometry['center']
                if wz > self.max_detection_depth_m:
                    continue

                diagonal_mm = camera_geometry['diagonal_m'] * 1000.0
                length_mm = camera_geometry['length_m'] * 1000.0
                width_mm = camera_geometry['width_m'] * 1000.0
                severity = self.classify_crack(diagonal_mm)
                marker_color = self.severity_color(severity)

                try:
                    p = PointStamped()
                    p.header.frame_id = source_frame
                    p.header.stamp = img_msg.header.stamp
                    p.point.x = float(wx)
                    p.point.y = float(wy)
                    p.point.z = float(wz)

                    # a) 이미지가 찍힌 시각의 카메라 -> odom 변환을 찾음
                    transform = self.lookup_camera_transform(
                        camera_name,
                        source_frame,
                        img_msg.header.stamp
                    )
                    # b) 찾은 변환 행렬을 점에 직접 곱해줌
                    world_p = do_transform_point(p, transform)

                    # 4. 반원통 전개도 매핑
                    # odom은 로봇 스폰 위치를 원점으로 쓰므로
                    # 터널 월드 좌표로 보정한다.
                    world_x = world_p.point.x + self.odom_origin_world_x
                    world_y = world_p.point.y + self.odom_origin_world_y
                    world_z = world_p.point.z + self.odom_origin_world_z
                    u = (world_x - self.tunnel_x_min) / self.tunnel_length
                    if u < 0.0 or u > 1.0:
                        continue
                    distance_from_start_m = world_x - self.tunnel_x_min

                    theta = math.atan2(world_z, world_y)
                    v = max(0, min(1, theta / math.pi))

                    px, py = self.map_uv_to_pixel(u, v)

                    # 전개도는 같은 균열로 보이는 반복 탐지를 cluster로 묶어 렌더링한다.
                    self.add_map_detection(
                        px,
                        py,
                        distance_from_start_m,
                        diagonal_mm,
                        length_mm,
                        width_mm
                    )
                    log_count = self.mapped_log_counts.get(camera_name, 0)
                    if log_count < 5:
                        self.get_logger().info(
                            f"{camera_name} mapped: "
                            f"world=({world_x:.2f}, "
                            f"{world_y:.2f}, {world_z:.2f}), "
                            f"uv=({u:.2f}, {v:.2f}), pixel=({px}, {py}), "
                            f"x_from_start={distance_from_start_m:.2f}m, "
                            f"size=({length_mm:.0f}x{width_mm:.0f}mm, "
                            f"diag={diagonal_mm:.0f}mm), "
                            f"class={severity}, "
                            f"source={camera_geometry['source']}"
                        )
                        self.mapped_log_counts[camera_name] = log_count + 1

                    cv2.rectangle(cv_img, (u1, v1), (u2, v2), marker_color, 2)
                    label = (
                        f"{camera_name}: {severity} "
                        f"{diagonal_mm:.0f}mm"
                    )
                    cv2.putText(
                        cv_img,
                        label,
                        (u1, max(v1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        marker_color,
                        2
                    )

                except Exception as e:
                    self.get_logger().warn(f"{camera_name} camera TF 에러: {e}")
                    continue

        cv2.imshow(f"{camera_name.capitalize()} Camera", cv_img)
        cv2.imshow("Tunnel Unrolled Map", self.unrolled_map)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CrackDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.imwrite("final_tunnel_inspection_map.png", node.unrolled_map)
        node.get_logger().info("최종 맵 저장 완료!")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

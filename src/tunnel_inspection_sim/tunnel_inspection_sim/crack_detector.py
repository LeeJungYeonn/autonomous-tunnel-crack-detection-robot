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
# 🚨 버그 해결을 위한 명시적 변환 함수 임포트
import tf2_geometry_msgs 
from tf2_geometry_msgs import do_transform_point

class CrackDetectorNode(Node):
    def __init__(self):
        super().__init__('crack_detector_node')
        self.bridge = CvBridge()
        
        # [신규 모델 적용] Hugging Face에서 다운받은 모델 경로
        model_path = os.path.expanduser('~/tunnel_ws/src/tunnel_inspection_sim/models/yolov8_crack_seg.pt')
        self.get_logger().info(f"YOLO 모델 로딩: {model_path}")
        self.yolo_model = YOLO(model_path) 
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 양쪽 RGB-D 카메라를 각각 동기화해서 같은 탐지/매핑 파이프라인으로 처리한다.
        self.camera_subscribers = []
        self.camera_synchronizers = []
        self.logged_camera_frames = set()
        self.mapped_log_counts = {}
        self.camera_names = ('left', 'right', 'top')
        for camera_name in self.camera_names:
            self.create_camera_synchronizer(camera_name)
        
        # [시각화 캔버스 설정]
        self.map_w, self.map_h = 1000, 300
        self.tunnel_x_min = float(self.declare_parameter('tunnel_x_min', -5.0).value)
        self.tunnel_x_max = float(self.declare_parameter('tunnel_x_max', 5.0).value)
        self.odom_origin_world_x = float(self.declare_parameter('odom_origin_world_x', -4.0).value)
        self.tunnel_length = self.tunnel_x_max - self.tunnel_x_min
        if self.tunnel_length <= 0.0:
            self.get_logger().warn("터널 x 범위가 잘못되어 기본값(-5.0~5.0)을 사용합니다.")
            self.tunnel_x_min = -5.0
            self.tunnel_x_max = 5.0
            self.tunnel_length = 10.0

        self.unrolled_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 255
        self.get_logger().info(
            "✅ 시스템 준비 완료! "
            f"(터널 X: {self.tunnel_x_min:.1f}~{self.tunnel_x_max:.1f}, "
            f"odom 원점 월드 X: {self.odom_origin_world_x:.1f}, "
            f"카메라: {'/'.join(self.camera_names)})"
        )

    def create_camera_synchronizer(self, camera_name):
        topic_prefix = f'/{camera_name}_camera'
        img_sub = message_filters.Subscriber(self, Image, f'{topic_prefix}/image')
        depth_sub = message_filters.Subscriber(self, Image, f'{topic_prefix}/depth')
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
            lambda img_msg, depth_msg, info_msg, name=camera_name: self.sync_callback(
                name,
                img_msg,
                depth_msg,
                info_msg
            )
        )
        self.camera_subscribers.extend([img_sub, depth_sub, info_sub])
        self.camera_synchronizers.append(synchronizer)

    def sync_callback(self, camera_name, img_msg, depth_msg, info_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        source_frame = f'{camera_name}_camera_optical_frame'
        reported_frame = info_msg.header.frame_id
        if camera_name not in self.logged_camera_frames:
            if reported_frame and reported_frame != source_frame:
                self.get_logger().warn(
                    f"{camera_name} camera_info frame_id는 '{reported_frame}'이지만 "
                    f"depth 좌표는 '{source_frame}'로 변환합니다."
                )
            else:
                self.get_logger().info(
                    f"{camera_name} camera depth 좌표 frame: {source_frame}"
                )
            self.logged_camera_frames.add(camera_name)
        
        fx, fy, cx, cy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]
        
        # 모델 추론
        results = self.yolo_model(cv_img, conf=0.2, verbose=False)
        
        for r in results:
            # Segmentation 모델이어도 BBox(네모 박스)는 동일하게 추출 가능!
            for box in r.boxes:
                u1, v1, u2, v2 = map(int, box.xyxy[0])
                cu, cv = (u1 + u2) // 2, (v1 + v2) // 2
                
                # 1. 탐지 즉시 파란색 박스 그리기
                cv2.rectangle(cv_img, (u1, v1), (u2, v2), (255, 0, 0), 2)

                # 2. Median Depth (노이즈 필터링)
                patch = cv_depth[max(0,cv-5):min(cv_depth.shape[0],cv+5), max(0,cu-5):min(cv_depth.shape[1],cu+5)]
                valid_depths = patch[patch > 0]
                
                if len(valid_depths) == 0: continue
                Z = np.median(valid_depths)
                if np.isnan(Z) or Z < 0.1: continue

                # 3. 3D 좌표 계산 (카메라 기준)
                wx, wy, wz = (cu - cx) * Z / fx, (cv - cy) * Z / fy, Z
                
                try:
                    p = PointStamped()
                    p.header.frame_id = source_frame
                    p.header.stamp = img_msg.header.stamp
                    p.point.x, p.point.y, p.point.z = float(wx), float(wy), float(wz)
                    
                    # 🚨 확실한 TF 변환 로직 (버그 해결) 🚨
                    # a) 먼저 카메라 -> odom(절대 좌표계)까지의 변환 행렬을 찾음
                    transform = self.tf_buffer.lookup_transform(
                        'odom', 
                        source_frame,
                        rclpy.time.Time() # 가장 최신 TF 사용
                    )
                    # b) 찾은 변환 행렬을 점에 직접 곱해줌
                    world_p = do_transform_point(p, transform)
                    
                    # 4. 반원통 전개도 매핑
                    # odom은 로봇 스폰 위치를 원점으로 쓰므로, 터널 월드 X로 보정한다.
                    world_x = world_p.point.x + self.odom_origin_world_x
                    u = (world_x - self.tunnel_x_min) / self.tunnel_length
                    if u < 0.0 or u > 1.0:
                        continue

                    theta = math.atan2(world_p.point.z, world_p.point.y) 
                    v = max(0, min(1, theta / math.pi))
                    
                    px = int(round(u * (self.map_w - 1)))
                    py = int(round(v * (self.map_h - 1)))
                    px = max(0, min(self.map_w - 1, px))
                    py = max(0, min(self.map_h - 1, py))
                    
                    # 전개도에 핀 마커(빨간점) 찍기
                    cv2.circle(self.unrolled_map, (px, py), 5, (0, 0, 255), -1)
                    log_count = self.mapped_log_counts.get(camera_name, 0)
                    if log_count < 5:
                        self.get_logger().info(
                            f"{camera_name} mapped: "
                            f"world=({world_x:.2f}, {world_p.point.y:.2f}, {world_p.point.z:.2f}), "
                            f"uv=({u:.2f}, {v:.2f}), pixel=({px}, {py})"
                        )
                        self.mapped_log_counts[camera_name] = log_count + 1
                    
                    # 매핑 성공 시 초록색 박스로 변경!
                    cv2.rectangle(cv_img, (u1, v1), (u2, v2), (0, 255, 0), 2)
                    label = f"{camera_name}: Mapped!"
                    cv2.putText(
                        cv_img,
                        label,
                        (u1, max(v1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
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
        node.get_logger().info("💾 최종 맵 저장 완료!")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

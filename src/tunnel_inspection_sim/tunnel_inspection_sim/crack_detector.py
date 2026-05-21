import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import math
from ultralytics import YOLO
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped

class CrackDetectorNode(Node):
    def __init__(self):
        super().__init__('crack_detector_node')
        self.bridge = CvBridge()
        
        # [모델 로드] 첨부한 crack_detector_model.pt 사용
        model_path = '/home/jen/tunnel_ws/src/tunnel_inspection_sim/models/crack_detector_model.pt'
        self.yolo_model = YOLO(model_path) 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 동기화 구독
        img_sub = message_filters.Subscriber(self, Image, '/left_camera/image')
        depth_sub = message_filters.Subscriber(self, Image, '/left_camera/depth')
        info_sub = message_filters.Subscriber(self, CameraInfo, '/left_camera/camera_info')
        self.ts = message_filters.ApproximateTimeSynchronizer([img_sub, depth_sub, info_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)
        
        # [시각화 캔버스 설정] 10m 터널, 0.35m 반지름
        self.map_w, self.map_h = 1000, 300
        self.unrolled_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 255
        self.get_logger().info("✅ 시스템 준비 완료!")

    def sync_callback(self, img_msg, depth_msg, info_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        
        fx, fy, cx, cy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]
        
        # 모델 추론 (conf 낮춰서 균열 놓치지 않게)
        results = self.yolo_model(cv_img, conf=0.01, verbose=True)
        
        for r in results:
            for box in r.boxes:
                u1, v1, u2, v2 = map(int, box.xyxy[0])
                cu, cv = (u1 + u2) // 2, (v1 + v2) // 2
                
                # Median Depth (노이즈 필터링)
                patch = cv_depth[max(0,cv-5):min(cv_depth.shape[0],cv+5), max(0,cu-5):min(cv_depth.shape[1],cu+5)]
                Z = np.median(patch[patch > 0])
                
                if np.isnan(Z) or Z < 0.1: continue

                # 3D 좌표 변환
                wx, wy, wz = (cu - cx) * Z / fx, (cv - cy) * Z / fy, Z
                
                # TF 좌표 변환 (World 좌표계로)
                try:
                    p = PointStamped()
                    p.header.frame_id = info_msg.header.frame_id
                    p.header.stamp = img_msg.header.stamp
                    p.point.x, p.point.y, p.point.z = wx, wy, wz
                    world_p = self.tf_buffer.transform(p, 'odom')
                    
                    # [매핑 수식] 반원통 전개
                    # u = x / tunnel_length, v = theta / pi
                    u = (world_p.point.x + 5.0) / 10.0
                    theta = math.atan2(world_p.point.z, world_p.point.y) # 반원통 y, z좌표
                    v = max(0, min(1, theta / math.pi))
                    
                    px, py = int(u * self.map_w), int((1.0 - v) * self.map_h)
                    
                    # 마커 찍기 (안전: 녹색, 위험: 빨강)
                    cv2.circle(self.unrolled_map, (px, py), 5, (0, 0, 255), -1)
                except: continue

                cv2.rectangle(cv_img, (u1, v1), (u2, v2), (0, 0, 255), 2)
        
        cv2.imshow("Camera", cv_img)
        cv2.imshow("Tunnel Unrolled Map", self.unrolled_map)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CrackDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 노드 종료 시 최종 전개도 이미지 저장
        cv2.imwrite("final_tunnel_inspection_map.png", node.unrolled_map)
        node.get_logger().info("💾 최종 진단 전개도가 'final_tunnel_inspection_map.png'로 저장되었습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
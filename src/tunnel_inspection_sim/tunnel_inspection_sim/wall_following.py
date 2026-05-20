import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class WallFollowerNode(Node):
    def __init__(self):
        super().__init__('wall_follower_node')
        
        # 1. 속도 제어 퍼블리셔
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 2. LiDAR 데이터 구독자
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        # 3. PID 제어 게인 (시뮬레이터에서 테스트했던 최적의 값 세팅)
        self.kp = 0.8
        self.ki = 0.0
        self.kd = 0.3
        
        self.prev_error = 0.0
        self.error_sum = 0.0
        
        self.is_running = True
        self.get_logger().info("🚀 LiDAR 기반 Wall Following 주행을 시작합니다!")

    def get_average_distance(self, ranges, center_idx, window):
        """특정 각도 주변의 레이저 거리 평균을 구하는 함수 (inf, nan 예외 처리)"""
        start = max(0, center_idx - window)
        end = min(len(ranges), center_idx + window)
        
        valid_ranges = [r for r in ranges[start:end] if not math.isinf(r) and not math.isnan(r) and r > 0.0]
        
        # 유효한 값이 없으면 최대 사거리(10m) 반환
        if not valid_ranges:
            return 10.0
        return sum(valid_ranges) / len(valid_ranges)

    def scan_callback(self, msg):
        if not self.is_running:
            return

        # Waffle 모델 LiDAR 기준: 0번(-180도/후방), 90번(-90도/우측), 180번(0도/정면), 270번(+90도/좌측)
        ranges = msg.ranges
        
        # 전방, 좌측, 우측의 평균 거리 계산 (각각 +- 10도 범위)
        d_front = self.get_average_distance(ranges, 180, 10)
        d_left = self.get_average_distance(ranges, 270, 10)
        d_right = self.get_average_distance(ranges, 90, 10)

        twist = Twist()

        # [종료 조건] 1. 전방에 0.4m 이내 장애물 감지 / 2. 양쪽 벽 거리가 1.5m 이상 (터널 탈출)
        if d_front < 0.4 or (d_left + d_right) > 1.5:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(twist)
            self.get_logger().info(f'🛑 정지 조건 충족 (전방: {d_front:.2f}m, 좌우 합: {d_left+d_right:.2f}m). 주행을 종료합니다.')
            self.is_running = False
            return

        # [제어 로직] 중앙 오차 (e = d_left - d_right)
        # 오차가 양수(좌측이 멂 = 우측에 치우침) -> 좌회전(양수 angular.z) 필요!
        error = d_left - d_right
        
        # PID 연산
        self.error_sum += error
        error_diff = error - self.prev_error
        
        angular_z = (self.kp * error) + (self.ki * self.error_sum) + (self.kd * error_diff)
        
        # 회전 속도 제한 (너무 급격하게 도는 것 방지)
        angular_z = max(-1.0, min(1.0, angular_z))
        
        self.prev_error = error

        # 선속도는 일정하게 유지 (초당 0.15m)
        twist.linear.x = 0.15
        twist.angular.z = angular_z

        self.publisher_.publish(twist)
        
        # 로그 출력 (확인용)
        self.get_logger().info(f'L: {d_left:.2f}m | R: {d_right:.2f}m | Err: {error:.2f} | Ang_z: {angular_z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = WallFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class SimpleDriveNode(Node):
    def __init__(self):
        super().__init__('simple_drive_node')
        
        # 1. 속도 명령 퍼블리셔 (/cmd_vel)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 2. 현재 위치 구독자 (/odom)
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.target_x = 10.0  # 출발지점부터 10미터를 이동해야 "월드" X=5.0에 도달함
        self.goal_reached = False
        self.get_logger().info("🚀 자율 직진 주행 노드를 시작합니다! (목표: X = 5.0)")

    def odom_callback(self, msg):
        if self.goal_reached:
            return

        # 현재 X 좌표 추출
        current_x = msg.pose.pose.position.x
        twist = Twist()

        # 현재 위치가 목표치보다 작으면 전진
        if current_x < self.target_x:
            twist.linear.x = 0.5  # 초당 0.5m 속도로 전진
            # 터미널 창에 너무 많은 로그가 뜨지 않게 소수점 2자리까지만 출력
            self.get_logger().info(f'직진 중... 현재 X: {current_x:.2f}m / {self.target_x}m')
        else:
            # 목표치 도달 시 정지
            twist.linear.x = 0.0
            self.goal_reached = True
            self.get_logger().info('✅ 목표 지점(X=5.0) 도달! 주행을 종료합니다.')

        # 바퀴에 명령 전달
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleDriveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
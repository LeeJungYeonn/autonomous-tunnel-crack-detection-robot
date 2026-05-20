import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_name = 'tunnel_inspection_sim'
    pkg_share = get_package_share_directory(pkg_name)

    # 1. Xacro 파일 읽기 및 변환
    xacro_file = os.path.join(pkg_share, 'urdf', 'tunnel_robot.xacro')
    doc = xacro.process_file(xacro_file)
    urdf_xml = doc.toxml()

    # 🌟 핵심 해결책: 변환된 로봇 모델을 임시 파일로 물리적으로 저장 🌟
    urdf_path = os.path.join(tempfile.gettempdir(), 'tunnel_robot.urdf')
    with open(urdf_path, 'w') as f:
        f.write(urdf_xml)

    # 2. Robot State Publisher (TF 퍼블리시)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml}]
    )

    # 3. Gazebo에 로봇 스폰 (-string 대신 가장 안정적인 -file 옵션 사용!)
    spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'turtlebot3_tunnel',
            '-file', urdf_path,         # 저장된 파일 경로를 가제보에 직접 던져줌
            '-x', '-4.0', '-y', '0.0', '-z', '0.09' 
        ],
        output='screen'
    )

    # 4. ROS-Gazebo 브릿지 실행
    bridge_config = os.path.join(pkg_share, 'config', 'bridge.yaml')
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_config,
            'qos_overrides./tf_static.publisher.durability': 'transient_local'
        }],
        output='screen'
    )

    return LaunchDescription([
        rsp_node,
        spawn_node,
        bridge_node
    ])
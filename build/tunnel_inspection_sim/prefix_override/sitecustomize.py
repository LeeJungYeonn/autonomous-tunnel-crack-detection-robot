import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/leejungyeon/ros2_ws/install/tunnel_inspection_sim'

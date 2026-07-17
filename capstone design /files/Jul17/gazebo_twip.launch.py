import os
import re
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('cart_pendulum_sim')
    urdf_path = os.path.join(pkg_share, 'urdf', 'twip.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    robot_description = robot_description.replace(
        '$(find cart_pendulum_sim)', pkg_share
    )
    robot_description = re.sub(r'<!--.*?-->', '', robot_description, flags=re.DOTALL)
    robot_description = re.sub(r'\s+', ' ', robot_description).strip()

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'twip'],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    wheel_effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wheel_effort_controller'],
        output='screen',
    )

    delayed_controllers = TimerAction(
        period=3.0,
        actions=[joint_state_broadcaster_spawner, wheel_effort_controller_spawner],
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        delayed_controllers,
    ])
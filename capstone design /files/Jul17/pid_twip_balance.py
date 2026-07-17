#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class PidTwipBalance(Node):
    def __init__(self):
        super().__init__('pid_twip_balance')

        # ===== PID 게인 (시작값, 튜닝 필요) =====
        self.kp = 30.0
        self.ki = 0.0
        self.kd = 3.0

        # ===== 내부 상태 =====
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()

        self.torso_angle = 0.0

        self.max_effort = 10.0  # URDF의 effort limit과 동일

        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/wheel_effort_controller/commands',
            10
        )

        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('PID TWIP Balance 시작됨.')

    def joint_state_callback(self, msg: JointState):
        try:
            idx = msg.name.index('torso_joint')
            angle = msg.position[idx]
            # -pi ~ pi 범위로 정규화
            self.torso_angle = np.arctan2(np.sin(angle), np.cos(angle))
        except ValueError:
            pass

    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        # 목표: 각도 0 (수직)
        error = 0.0 - self.torso_angle

        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        effort = p_term + i_term + d_term
        effort = max(-self.max_effort, min(self.max_effort, effort))

        # 양쪽 바퀴에 동일한 토크 (좌우 대칭, 앞뒤 균형만 신경씀)
        msg = Float64MultiArray()
        msg.data = [effort, effort]
        self.pub.publish(msg)

        self.prev_error = error
        self.prev_time = now


def main(args=None):
    rclpy.init(args=args)
    node = PidTwipBalance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
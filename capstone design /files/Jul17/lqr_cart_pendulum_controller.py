#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class LqrCartPendulumController(Node):
    def __init__(self):
        super().__init__('lqr_cart_pendulum_controller')

        # ===== LQR 게인 (아까 계산한 값 그대로) =====
        # u = -K @ [x, x_dot, theta, theta_dot]
        self.K = np.array([-3.16227766, -5.83787482, -48.14263393, -11.83502798])

        # ===== 힘 제한 =====
        self.max_effort = 50.0

        # ===== 현재 상태 저장용 변수 =====
        self.cart_pos = 0.0
        self.cart_vel = 0.0
        self.pole_angle = 0.0
        self.pole_angvel = 0.0

        # 구독: 조인트 상태
        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # 발행: 힘 명령
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/cart_effort_controller/commands',
            10
        )

        # 50Hz 제어 루프
        self.timer = self.create_timer(0.02, self.control_loop)al

        self.get_logger().info('LQR Cart Pendulum Controller 시작됨.')

    def joint_state_callback(self, msg: JointState):
        try:
            cart_idx = msg.name.index('cart_joint')
            pole_idx = msg.name.index('pole_joint')

            self.cart_pos = msg.position[cart_idx]
            self.cart_vel = msg.velocity[cart_idx]

            # 각도를 -pi ~ pi 범위로 정규화 (continuous joint라 값이 계속 누적될 수 있음)
            angle = msg.position[pole_idx]
            self.pole_angle = np.arctan2(np.sin(angle), np.cos(angle))
            self.pole_angvel = msg.velocity[pole_idx]

        except ValueError:
            pass

    def control_loop(self):
        state = np.array([self.cart_pos, self.cart_vel, self.pole_angle, self.pole_angvel])

        # LQR 제어법칙: u = -K @ state
        effort = -float(self.K @ state)

        # 힘 제한
        effort = max(-self.max_effort, min(self.max_effort, effort))

        msg = Float64MultiArray()
        msg.data = [effort]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LqrCartPendulumController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
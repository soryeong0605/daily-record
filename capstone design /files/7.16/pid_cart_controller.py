#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState


class PidCartController(Node):
    def __init__(self):
        super().__init__('pid_cart_controller')

        # ===== 목표 위치 =====
        self.target_position = 1.0  # 카트가 도달할 목표 x위치 (m)

        # ===== PID 게인 (일단 시작값, 나중에 튜닝) =====
        self.kp = 20.0
        self.ki = 0.5
        self.kd = 10.0

        # ===== 내부 상태 변수 =====
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()

        # ===== 현재 카트 위치 (joint_states에서 갱신) =====
        self.cart_position = 0.0

        # ===== 힘 제한 (URDF의 effort limit과 동일하게) =====
        self.max_effort = 50.0

        # 구독: 현재 조인트 상태 받기
        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # 발행: 힘 명령 보내기
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/cart_effort_controller/commands',
            10
        )

        # 목표 위치를 외부에서 받기 위한 구독 (새로 추가)
        self.target_sub = self.create_subscription(
            Float64,
            '/cart_target_position',
            self.target_callback,
            10
        )

        # 50Hz로 제어 루프 실행 (0.02초마다)
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('PID Cart Controller 시작됨. 목표 위치: %.2f m' % self.target_position)

    def joint_state_callback(self, msg: JointState):
        # joint_states 메시지에서 cart_joint의 위치를 찾아서 저장
        try:
            idx = msg.name.index('cart_joint')
            self.cart_position = msg.position[idx]
        except ValueError:
            pass  # 아직 cart_joint 정보가 안 왔으면 무시

    def target_callback(self, msg: Float64):
        self.target_position = msg.data
        self.get_logger().info('목표 위치 변경: %.2f m' % self.target_position)

    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        error = self.target_position - self.cart_position

        # P
        p_term = self.kp * error

        # I
        self.integral += error * dt
        i_term = self.ki * self.integral

        # D
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        effort = p_term + i_term + d_term

        # 힘 제한 (saturate)
        effort = max(-self.max_effort, min(self.max_effort, effort))

        msg = Float64MultiArray()
        msg.data = [effort]
        self.pub.publish(msg)

        self.prev_error = error
        self.prev_time = now


def main(args=None):
    rclpy.init(args=args)
    node = PidCartController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
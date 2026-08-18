# =========================
# Plot helpers (3D + 2D grid version)
# =========================
#
# 3D 워크스페이스 + pc(빨강) + opti(파랑)를 한눈에 보여주는 버전.
# 기존 2클래스(XY/XZ/YZ 평면 3개만 보여주던 것)랑 메서드 이름/인자가
# 완전히 똑같아서, flowbot.py 쪽 코드는 전혀 안 고쳐도 그대로 꽂힘.
#
# 레이아웃: 2x2 그리드
#   [3D 워크스페이스+점]   [XY]
#   [XZ]                   [YZ]

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

HULL_DOWNSAMPLE = 6  # 3D hull 그릴 때 점 솎아내는 비율 (성능용, 시각에만 영향)


class plot_helper:
    def setup_plot(self, points: np.ndarray, draw_hull: bool = True):
        points = np.asarray(points, dtype=float)
        P_vis = points[::max(1, int(HULL_DOWNSAMPLE)), :]

        mn = points.min(axis=0)
        mx = points.max(axis=0)
        center = 0.5 * (mn + mx)
        span = float((mx - mn).max())

        plt.ion()
        fig = plt.figure(figsize=(12, 10))

        # 2x2 grid: 3D at top-left, then XY top-right, XZ bottom-left, YZ bottom-right
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_yz = fig.add_subplot(2, 2, 4)

        # -------------------------
        # 3D workspace surface (convex hull)
        # -------------------------
        if draw_hull and len(P_vis) >= 4:
            try:
                hull3d = ConvexHull(P_vis)
                ax_3d.plot_trisurf(
                    P_vis[:, 0], P_vis[:, 1], P_vis[:, 2],
                    triangles=hull3d.simplices,
                    alpha=0.20,
                    linewidth=0.2,
                    edgecolor=(0.2, 0.2, 0.2, 0.25),
                )
            except Exception as e:
                print(f"[plot_helper] 3D hull 생성 실패 (그냥 점만 표시): {e}")

        # 실제 워크스페이스 점들 표시 (hull은 볼록껍질이라, 실제 도달불가
        # 영역(오목한 부분)까지는 안 보여주기 때문에 원본 점들을 같이 찍어야
        # 진짜 도달가능영역 형태를 눈으로 확인 가능). 배경(격자면)이 진해서
        # 안 보이던 문제 -> 배경 옅게, 점은 진하게/크게 조정.
        ax_3d.scatter(
            P_vis[:, 0], P_vis[:, 1], P_vis[:, 2],
            s=6, alpha=0.55, c="steelblue", label="workspace points",
        )

        # 3D 배경(격자면) 옅게 -- matplotlib 기본값이 회색으로 꽉 차 있어서
        # 옅은 점들을 가려버림. 배경을 거의 투명하게, 테두리선만 살짝 남김.
        for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))
        ax_3d.grid(True, alpha=0.15)

        ax_3d.set_title("3D Workspace + pc (red) + Opti (blue)")
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("z")

        ax_3d.set_xlim(center[0] - 0.5 * span, center[0] + 0.5 * span)
        ax_3d.set_ylim(center[1] - 0.5 * span, center[1] + 0.5 * span)
        ax_3d.set_zlim(center[2] - 0.5 * span, center[2] + 0.5 * span)
        ax_3d.invert_zaxis()   # Z축이 아래를 향하도록 (2D XZ/YZ 패널과 통일)

        # -------------------------
        # 2D hulls on each plane
        # -------------------------
        def _plot_hull_2d(ax, pts2):
            pts2 = np.asarray(pts2, dtype=float)
            if len(pts2) < 3:
                ax.scatter(pts2[:, 0], pts2[:, 1], s=2, alpha=0.3)
                return
            try:
                h = ConvexHull(pts2)
                poly = pts2[h.vertices]
                poly = np.vstack([poly, poly[0]])
                ax.plot(poly[:, 0], poly[:, 1], linewidth=1.0, alpha=0.6)
            except Exception:
                ax.scatter(pts2[:, 0], pts2[:, 1], s=2, alpha=0.3)

        ax_xy.set_title("XY"); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
        ax_xz.set_title("XZ"); ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
        ax_yz.set_title("YZ"); ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")

        for ax in (ax_xy, ax_xz, ax_yz):
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")

        if draw_hull:
            _plot_hull_2d(ax_xy, P_vis[:, [0, 1]])
            _plot_hull_2d(ax_xz, P_vis[:, [0, 2]])
            _plot_hull_2d(ax_yz, P_vis[:, [1, 2]])

        # -------------------------
        # 워크스페이스 경계점들 (PWM 0~26 스윕한 실제 점군, 작은 점으로)
        # -------------------------
        # convex hull 선은 볼록껍질 윤곽선만 보여주는 거라, 실제 점들을
        # 옅고 작게 같이 뿌려주면 오목한 부분까지 포함한 진짜 경계
        # 형태가 눈에 보임. highlight_pwm_corners()가 찍는 8개 중요
        # 꼭짓점(빨간 세모)이랑은 별개로, 배경처럼 옅게 깔림.
        ax_xy.scatter(P_vis[:, 0], P_vis[:, 1], s=3, alpha=0.35, c="steelblue", zorder=1)
        ax_xz.scatter(P_vis[:, 0], P_vis[:, 2], s=3, alpha=0.35, c="steelblue", zorder=1)
        ax_yz.scatter(P_vis[:, 1], P_vis[:, 2], s=3, alpha=0.35, c="steelblue", zorder=1)

        ax_xy.set_xlim(mn[0], mx[0]); ax_xy.set_ylim(mn[1], mx[1])
        ax_xz.set_xlim(mn[0], mx[0]); ax_xz.set_ylim(mn[2], mx[2])
        ax_yz.set_xlim(mn[1], mx[1]); ax_yz.set_ylim(mn[2], mx[2])
        ax_xz.invert_yaxis()
        ax_yz.invert_yaxis()

        # -------------------------
        # Handles (pc, opti, trail) for 3D + 2D
        # -------------------------
        p0 = P_vis[0] if len(P_vis) > 0 else np.array([0.0, 0.0, 0.0])

        # 3D scatters
        pc_3d = ax_3d.scatter([p0[0]], [p0[1]], [p0[2]], s=70, c="red", label="pc")
        opti_3d = ax_3d.scatter([p0[0]], [p0[1]], [p0[2]], s=55, c="blue", label="opti")
        (trail_3d,) = ax_3d.plot([], [], [], linewidth=1.0, alpha=0.9, label="opti trail")

        # 2D scatters
        pc_xy = ax_xy.scatter([p0[0]], [p0[1]], s=60, c="red", label="pc")
        pc_xz = ax_xz.scatter([p0[0]], [p0[2]], s=60, c="red", label="pc")
        pc_yz = ax_yz.scatter([p0[1]], [p0[2]], s=60, c="red", label="pc")

        opti_xy = ax_xy.scatter([p0[0]], [p0[1]], s=45, c="blue", label="opti")
        opti_xz = ax_xz.scatter([p0[0]], [p0[2]], s=45, c="blue", label="opti")
        opti_yz = ax_yz.scatter([p0[1]], [p0[2]], s=45, c="blue", label="opti")

        (trail_xy,) = ax_xy.plot([], [], linewidth=1.0, alpha=0.8, label="trail")
        (trail_xz,) = ax_xz.plot([], [], linewidth=1.0, alpha=0.8, label="trail")
        (trail_yz,) = ax_yz.plot([], [], linewidth=1.0, alpha=0.8, label="trail")

        ax_3d.legend(loc="best")
        ax_xy.legend(loc="best")
        ax_xz.legend(loc="best")
        ax_yz.legend(loc="best")

        fig.tight_layout()
        fig.show()
        plt.pause(0.001)

        axes = {"3d": ax_3d, "xy": ax_xy, "xz": ax_xz, "yz": ax_yz}
        pc_handles = {"3d": pc_3d, "xy": pc_xy, "xz": pc_xz, "yz": pc_yz}
        opti_handles = {"3d": opti_3d, "xy": opti_xy, "xz": opti_xz, "yz": opti_yz}
        trail_handles = {"3d": trail_3d, "xy": trail_xy, "xz": trail_xz, "yz": trail_yz}

        return fig, axes, pc_handles, opti_handles, trail_handles

    def update_point_handle(self, pc_handles, pc: np.ndarray):
        pc = np.asarray(pc, dtype=float).reshape(3,)

        # 3D
        pc_handles["3d"]._offsets3d = ([pc[0]], [pc[1]], [pc[2]])

        # 2D: set_offsets expects Nx2
        pc_handles["xy"].set_offsets([[pc[0], pc[1]]])
        pc_handles["xz"].set_offsets([[pc[0], pc[2]]])
        pc_handles["yz"].set_offsets([[pc[1], pc[2]]])

    def update_opti_handle(self, opti_handles, p: np.ndarray):
        p = np.asarray(p, dtype=float).reshape(3,)

        opti_handles["3d"]._offsets3d = ([p[0]], [p[1]], [p[2]])
        opti_handles["xy"].set_offsets([[p[0], p[1]]])
        opti_handles["xz"].set_offsets([[p[0], p[2]]])
        opti_handles["yz"].set_offsets([[p[1], p[2]]])

    def highlight_pwm_corners(self, axes, robot, pwm_max: float = 26.0, pwm_min: float = 0.0):
        """
        run_workspace_linear.py에서 했던 것처럼, PWM 0/pwm_max 조합 8개
        (꼭짓점들)의 실제 도달 위치를 계산해서 3D + 2D 패널에 빨간 점 +
        라벨로 표시. setup_plot() 호출 직후, 딱 한 번만 부르면 됨.

        사용 예 (teleop.py / flowbot.py 쪽에서):
            fb.pl.highlight_pwm_corners(fb.axes, fb.flowbot, pwm_max=26)
        """
        combos = [
            (pwm_min, pwm_min, pwm_min),
            (pwm_max, pwm_min, pwm_min),
            (pwm_min, pwm_max, pwm_min),
            (pwm_min, pwm_min, pwm_max),
            (pwm_max, pwm_max, pwm_min),
            (pwm_min, pwm_max, pwm_max),
            (pwm_max, pwm_min, pwm_max),
            (pwm_max, pwm_max, pwm_max),
        ]

        pts = []
        for pwm_tuple in combos:
            pwm = np.array(pwm_tuple, dtype=float)
            pb = robot.pwm_to_pressure(pwm)
            fk = robot.forward_kinematics_from_pressures(pb)
            pc = np.asarray(fk["pc"], dtype=float).reshape(3,)
            pts.append(pc)
            print(f"[plot_helper] PWM={pwm_tuple} -> pc={np.round(pc, 2)}")

        pts = np.array(pts)

        ax_3d = axes["3d"]
        ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=90, c="red", marker="^",
                      label="PWM corners", zorder=5)
        for pwm_tuple, pc in zip(combos, pts):
            ax_3d.text(pc[0], pc[1], pc[2], f"{pwm_tuple}", size=7, color="darkred")

        for key, idx in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            ax = axes[key]
            i, j = idx
            ax.scatter(pts[:, i], pts[:, j], s=60, c="red", marker="^",
                      label="PWM corners", zorder=5)

        ax_3d.legend(loc="best")

    def update_trail_handle(self, trail_handles, trail_xyz: np.ndarray):
        if trail_xyz is None or len(trail_xyz) == 0:
            trail_handles["3d"].set_data([], [])
            trail_handles["3d"].set_3d_properties([])

            trail_handles["xy"].set_data([], [])
            trail_handles["xz"].set_data([], [])
            trail_handles["yz"].set_data([], [])
            return

        trail_xyz = np.asarray(trail_xyz, dtype=float)

        # 3D line
        trail_handles["3d"].set_data(trail_xyz[:, 0], trail_xyz[:, 1])
        trail_handles["3d"].set_3d_properties(trail_xyz[:, 2])

        # 2D projections
        trail_handles["xy"].set_data(trail_xyz[:, 0], trail_xyz[:, 1])
        trail_handles["xz"].set_data(trail_xyz[:, 0], trail_xyz[:, 2])
        trail_handles["yz"].set_data(trail_xyz[:, 1], trail_xyz[:, 2])

"""
show_zarr_timeline.py

zarr 에피소드를 "초 단위로 어떤 값이 찍혔는지" CSV처럼 훑어보는 스크립트.
- --interval_sec 간격으로 샘플링해서 보여줌(기본 1초)
- dt(직전 스텝과의 시간차)가 비정상적으로 크면 [GAP!] 표시로 따로 강조
- --csv_out 지정하면 전체 스텝(샘플링 없이) CSV로도 저장

사용법:
    # 터미널에 1초 간격으로 훑어보기
    python show_zarr_timeline.py --zarr_path data/demo_data/dataset.zarr --episode 0

    # 전체 스텝을 CSV로도 뽑기
    python show_zarr_timeline.py --zarr_path data/demo_data/dataset.zarr --episode 0 --csv_out timeline_ep0.csv
"""

import argparse
import csv as csv_module

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", default="data/demo_data/dataset.zarr")
    parser.add_argument("--episode", type=int, default=0, help="몇 번째 에피소드를 볼지 (0부터)")
    parser.add_argument("--interval_sec", type=float, default=1.0,
                         help="터미널 출력 샘플링 간격(초). 0이면 전체 스텝 다 출력")
    parser.add_argument("--gap_threshold_ms", type=float, default=100.0,
                         help="이 값(ms)보다 dt가 크면 [GAP!]으로 표시")
    parser.add_argument("--csv_out", default=None, help="지정하면 전체 스텝을 이 경로에 CSV로 저장")
    args = parser.parse_args()

    root = zarr.open(args.zarr_path, mode="r")
    data = root["data"]
    ends = np.array(root["meta"]["episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])

    if args.episode >= len(ends):
        print(f"에피소드 {args.episode}번은 없습니다 (총 {len(ends)}개, 0~{len(ends)-1})")
        return

    s, e = int(starts[args.episode]), int(ends[args.episode])
    ts = np.array(data["timestamp"][s:e])
    op_mode = np.array(data["operation_mode"][s:e])
    franka_pose = np.array(data["franka_eef_pose"][s:e])
    pwm = np.array(data["pwm_signals"][s:e])
    action = np.array(data["action"][s:e])

    elapsed = ts - ts[0]
    dt_ms = np.concatenate([[0.0], np.diff(ts) * 1000.0])
    mode_label = np.where(
        op_mode[:, 0] == 1,
        np.where(op_mode[:, 1] == 1, "release", "franka"),
        "gripper",
    )

    print(f"\n에피소드 {args.episode}: {e - s} 스텝, 총 {elapsed[-1]:.1f}초")
    print("-" * 90)
    header = f"{'step':>5} {'elapsed(s)':>10} {'dt(ms)':>8} {'mode':>8} {'franka_xyz':>28} {'pwm':>16}"
    print(header)
    print("-" * 90)

    next_mark = 0.0
    for i in range(len(ts)):
        is_gap = dt_ms[i] > args.gap_threshold_ms
        show_by_interval = args.interval_sec <= 0 or elapsed[i] >= next_mark
        if not (show_by_interval or is_gap):
            continue
        if show_by_interval:
            next_mark += args.interval_sec

        xyz_str = f"[{franka_pose[i,0]:.3f},{franka_pose[i,1]:.3f},{franka_pose[i,2]:.3f}]"
        pwm_str = f"[{pwm[i,0]:.0f},{pwm[i,1]:.0f},{pwm[i,2]:.0f}]"
        gap_tag = "  <== [GAP!]" if is_gap else ""
        print(f"{i:>5} {elapsed[i]:>10.2f} {dt_ms[i]:>8.1f} {mode_label[i]:>8} "
              f"{xyz_str:>28} {pwm_str:>16}{gap_tag}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow([
                "step", "elapsed_sec", "dt_ms", "mode",
                "franka_x", "franka_y", "franka_z",
                "pwm1", "pwm2", "pwm3",
                "action_vx", "action_vy", "action_vz", "action_wz",
                "action_pwm1", "action_pwm2", "action_pwm3",
            ])
            for i in range(len(ts)):
                writer.writerow([
                    i, f"{elapsed[i]:.4f}", f"{dt_ms[i]:.2f}", mode_label[i],
                    *[f"{v:.5f}" for v in franka_pose[i, :3]],
                    *[f"{v:.3f}" for v in pwm[i]],
                    *[f"{v:.5f}" for v in action[i]],
                ])
        print(f"\n전체 {e-s}스텝 CSV -> {args.csv_out}")


if __name__ == "__main__":
    main()

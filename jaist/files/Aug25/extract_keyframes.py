"""
extract_keyframes.py - franka_soryeong/combined_teleop/extract_keyframes.py

teleop 실시간 코드(main.py, logger_task.py 등)와 완전히 분리된 오프라인
스크립트. 이미 저장된 episodes/episode_XXXX_.../ 를 읽기만 하고, 원본
파일은 절대 건드리지 않음.

에피소드 CSV의 'mode' 컬럼(franka / flowbot / flowbot_release)이 바뀌는
지점(구간 시작/끝) + 에피소드 전체의 첫/마지막 프레임을 "키프레임"으로
뽑아서, camera_0/camera_1 둘 다 별도 폴더에 리사이즈해서 저장함.

목적: 나중에 이 키프레임들만 API(비전 모델)에 보내서 에피소드 성공
여부를 판단하기 위함 -- 원본 수천 장을 다 보낼 필요 없이, 의미있는
전환 순간 몇 장이면 충분하다는 전제.

사용법:
    # episodes/ 폴더 안의 모든 에피소드 처리
    python extract_keyframes.py

    # 특정 폴더/출력 경로 지정
    python extract_keyframes.py --episodes_dir episodes --output_dir episodes/keyframes

    # 리사이즈 크기/화질 조절
    python extract_keyframes.py --max_side 480 --quality 85

    # 특정 에피소드 하나만
    python extract_keyframes.py --episode episode_0003_20260825_170738
"""

import argparse
import csv as csv_module
import glob
import json
import os

import cv2


def find_episode_csvs(episodes_dir, episode_filter=None):
    """episodes_dir 안의 episode_*.csv 파일들을 (base_name, csv_path) 쌍으로 반환."""
    pattern = os.path.join(episodes_dir, "episode_*.csv")
    results = []
    for csv_path in sorted(glob.glob(pattern)):
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        if episode_filter and base_name != episode_filter:
            continue
        results.append((base_name, csv_path))
    return results


def load_episode_rows(csv_path):
    """CSV를 읽어서 [{"frame_idx": int, "mode": str, "timestamp": str}, ...] 로 반환."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            rows.append({
                "frame_idx": int(row["frame_idx"]),
                "mode": row["mode"],
                "timestamp": row["timestamp"],
            })
    return rows


def select_keyframe_indices(rows):
    """
    mode 전환 지점(구간 시작 직후 프레임 + 구간 끝 직전 프레임) + 에피소드
    전체 첫/마지막 프레임을 뽑아서 frame_idx 오름차순으로 반환.
    각 frame_idx에 어떤 의미로 뽑혔는지(reason)도 같이 반환.
    """
    if not rows:
        return []

    selected = {}  # frame_idx -> reason (여러 이유 겹치면 마지막 것 우선)

    selected[rows[0]["frame_idx"]] = f"episode_start(mode={rows[0]['mode']})"
    selected[rows[-1]["frame_idx"]] = f"episode_end(mode={rows[-1]['mode']})"

    for i in range(1, len(rows)):
        prev_mode = rows[i - 1]["mode"]
        cur_mode = rows[i]["mode"]
        if cur_mode != prev_mode:
            # 구간이 바뀌는 순간: 직전 프레임(이전 구간의 끝) + 이 프레임(새 구간의 시작)
            selected[rows[i - 1]["frame_idx"]] = f"segment_end({prev_mode}->{cur_mode})"
            selected[rows[i]["frame_idx"]] = f"segment_start({prev_mode}->{cur_mode})"

    return sorted(selected.items())  # [(frame_idx, reason), ...]


def resize_and_save(src_path, dst_path, max_side):
    img = cv2.imread(src_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def process_episode(base_name, csv_path, episodes_dir, output_dir, camera_names, max_side):
    image_dir = os.path.join(episodes_dir, base_name)
    if not os.path.isdir(image_dir):
        print(f"[SKIP] {base_name}: 이미지 폴더 없음 ({image_dir})")
        return None

    rows = load_episode_rows(csv_path)
    keyframes = select_keyframe_indices(rows)
    if not keyframes:
        print(f"[SKIP] {base_name}: CSV에 행이 없음")
        return None

    row_by_idx = {r["frame_idx"]: r for r in rows}

    out_episode_dir = os.path.join(output_dir, base_name)
    manifest = []
    saved_count = 0
    missing_count = 0

    for frame_idx, reason in keyframes:
        row = row_by_idx.get(frame_idx, {})
        entry = {
            "frame_idx": frame_idx,
            "reason": reason,
            "mode": row.get("mode"),
            "timestamp": row.get("timestamp"),
            "cameras": {},
        }
        for cam_name in camera_names:
            src = os.path.join(image_dir, cam_name, f"{frame_idx:06d}.jpg")
            dst_dir = os.path.join(out_episode_dir, cam_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{frame_idx:06d}.jpg")
            if os.path.isfile(src):
                if resize_and_save(src, dst, max_side):
                    entry["cameras"][cam_name] = dst
                    saved_count += 1
                else:
                    missing_count += 1
            else:
                missing_count += 1
        manifest.append(entry)

    manifest_path = os.path.join(out_episode_dir, "manifest.json")
    os.makedirs(out_episode_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "episode": base_name,
            "total_rows": len(rows),
            "num_keyframes": len(keyframes),
            "frames": manifest,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] {base_name}: 키프레임 {len(keyframes)}개 지점 x 카메라 {len(camera_names)}대 "
          f"-> 저장 {saved_count}장, 누락 {missing_count}장  (manifest -> {manifest_path})")

    return {
        "episode": base_name,
        "num_keyframes": len(keyframes),
        "saved": saved_count,
        "missing": missing_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_dir", default="episodes", help="원본 에피소드(csv+이미지 폴더)가 있는 경로")
    parser.add_argument("--output_dir", default=None, help="키프레임 저장 경로 (기본: <episodes_dir>/keyframes)")
    parser.add_argument("--cameras", default="camera_0,camera_1", help="쉼표로 구분된 카메라 폴더명 목록")
    parser.add_argument("--max_side", type=int, default=480, help="리사이즈 시 긴 변 최대 길이(px)")
    parser.add_argument("--episode", default=None, help="특정 에피소드 base name(확장자 제외)만 처리")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.episodes_dir, "keyframes")
    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]

    episodes = find_episode_csvs(args.episodes_dir, episode_filter=args.episode)
    if not episodes:
        print(f"[extract_keyframes] {args.episodes_dir} 안에서 처리할 에피소드를 못 찾음.")
        return

    print(f"[extract_keyframes] {len(episodes)}개 에피소드 처리 시작 "
          f"(카메라: {camera_names}, max_side={args.max_side}px)\n")

    results = []
    for base_name, csv_path in episodes:
        result = process_episode(
            base_name, csv_path, args.episodes_dir, output_dir, camera_names, args.max_side
        )
        if result:
            results.append(result)

    total_saved = sum(r["saved"] for r in results)
    total_missing = sum(r["missing"] for r in results)
    print(f"\n[extract_keyframes] 완료: 에피소드 {len(results)}개, "
          f"총 저장 {total_saved}장, 누락 {total_missing}장")
    print(f"결과 위치: {output_dir}")


if __name__ == "__main__":
    main()

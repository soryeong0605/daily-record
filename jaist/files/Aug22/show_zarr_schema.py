"""
show_zarr_schema.py

zarr 데이터셋의 구조(그룹/배열 트리, 각 배열의 shape/dtype, attrs)만
깔끔하게 트리 형태로 출력.

사용법:
    python show_zarr_schema.py --zarr_path data/demo_data/dataset.zarr
"""

import argparse
import zarr


def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def print_node(node, name, indent=0):
    prefix = "  " * indent

    if isinstance(node, zarr.Group):
        attrs = dict(node.attrs)
        attrs_str = f"  attrs={attrs}" if attrs else ""
        print(f"{prefix}📁 {name}/{attrs_str}")
        for child_name in sorted(node.keys()):
            print_node(node[child_name], child_name, indent + 1)
    else:
        # array
        arr = node
        nbytes = arr.nbytes if hasattr(arr, "nbytes") else 0
        attrs = dict(arr.attrs)
        attrs_str = f"  attrs={attrs}" if attrs else ""
        print(
            f"{prefix}📄 {name:20s} shape={str(arr.shape):22s} "
            f"dtype={str(arr.dtype):10s} size={format_bytes(nbytes)}{attrs_str}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", default="data/demo_data/dataset.zarr")
    args = parser.parse_args()

    root = zarr.open(args.zarr_path, mode="r")

    print("=" * 70)
    print(f"zarr schema: {args.zarr_path}")
    print("=" * 70)
    print_node(root, "(root)")

    # 요약: 에피소드 수 / 총 스텝 수
    if "meta" in root and "episode_ends" in root["meta"]:
        import numpy as np
        episode_ends = np.array(root["meta"]["episode_ends"])
        print("\n" + "-" * 70)
        print(f"에피소드 수: {len(episode_ends)}")
        if len(episode_ends) > 0:
            print(f"총 스텝 수: {int(episode_ends[-1])}")
            lengths = np.diff(np.concatenate([[0], episode_ends]))
            print(f"에피소드별 길이: {lengths.tolist()}")


if __name__ == "__main__":
    main()

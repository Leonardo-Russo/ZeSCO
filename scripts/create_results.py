import argparse
import os
import re
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image


REQUIRED_VARIANTS = (
    "ground",
    "aerial_overlay",
    "distance_curve",
    "aerial_rays",
)


def find_available_sample_ids(results_dir: str) -> List[int]:
    """Discover sample IDs by scanning for files like sample_<id>_aerial_overlay.png."""
    ids = set()
    if not os.path.isdir(results_dir):
        return []
    pattern = re.compile(r"^sample_(\d+)_aerial_overlay\.png$")
    for name in os.listdir(results_dir):
        m = pattern.match(name)
        if m:
            try:
                ids.add(int(m.group(1)))
            except ValueError:
                pass
    return sorted(ids)


def build_paths_for_sample(results_dir: str, sample_id: int) -> List[str]:
    return [
        os.path.join(results_dir, f"sample_{sample_id}_{variant}.png")
        for variant in REQUIRED_VARIANTS
    ]


def ensure_all_exist(paths: List[str]) -> bool:
    return all(os.path.isfile(p) for p in paths)


def select_sample_ids(
    results_dir: str,
    indices: Optional[List[int]],
    num: int,
    use_random: bool,
) -> List[int]:
    available = find_available_sample_ids(results_dir)
    if not available:
        raise FileNotFoundError(f"No sample_*_aerial_overlay.png files found in {results_dir}")

    if indices:
        missing = [i for i in indices if i not in available]
        if missing:
            raise FileNotFoundError(
                f"Some requested sample IDs are not available in {results_dir}: {missing}\n"
                f"Available IDs: {available}"
            )
        chosen = indices
    else:
        candidates = [i for i in available if ensure_all_exist(build_paths_for_sample(results_dir, i))]
        if len(candidates) < num:
            raise FileNotFoundError(
                f"Only {len(candidates)} complete samples found (need {num})."
            )
        if use_random:
            chosen = random.sample(candidates, num)
        else:
            chosen = candidates[:num]

    # keep order stable
    return list(chosen)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_grid(
    results_dir: str,
    sample_ids: List[int],
    out_path: str,
    figsize: Tuple[int, int] = (12, 12),
    annotate_titles: bool = True,
) -> None:

    figsize = (12, 3*len(sample_ids))
    fig, axes = plt.subplots(len(sample_ids), 4, figsize=figsize)

    col_titles = [
        "Ground Image",
        "Aerial Image",
        "Distance over Orientation",
        "Candidate Directions",
    ]

    for row, sid in enumerate(sample_ids):
        paths = build_paths_for_sample(results_dir, sid)
        if not ensure_all_exist(paths):
            missing = [p for p in paths if not os.path.isfile(p)]
            raise FileNotFoundError(f"Missing images for sample {sid}: {missing}")

        for col, img_path in enumerate(paths):
            ax = axes[row, col]
            img = load_image(img_path)
            ax.imshow(img)
            ax.axis("off")
            if row == 0 and annotate_titles:
                ax.set_title(col_titles[col], fontsize=14, fontweight='bold')
        # annotate row label on the left
        axes[row, 0].set_ylabel(f"Sample {sid}", rotation=90, fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_indices(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip() != ""]
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise ValueError("--indices must be a comma-separated list of integers, e.g., 0,5,10,12")


def main(n_results=3):
    parser = argparse.ArgumentParser(description="Create a 4x4 results grid from per-sample images.")
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default=os.path.join("results", "cvg_bench"),
        help="Directory containing sample_*_{ground|aerial_overlay|distance_curve|aerial_rays}.png",
    )
    parser.add_argument(
        "--indices",
        "-i",
        type=str,
        default=None,
        help="Comma-separated sample IDs to include (exactly 4). If omitted, picks 4 automatically.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly choose 4 samples when --indices is not provided.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the grid image. Defaults to <results_dir>/grid_4x4.png",
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=12.0,
        help="Figure width in inches (height follows to keep square cells)",
    )
    args = parser.parse_args()

    indices = parse_indices(args.indices)

    sample_ids = select_sample_ids(
        args.results_dir,
        indices=indices,
        num=n_results,
        use_random=args.random,
    )

    out_path = args.output or os.path.join(args.results_dir, "results.png")
    figsize = (args.figwidth, args.figwidth)

    make_grid(
        results_dir=args.results_dir,
        sample_ids=sample_ids,
        out_path=out_path,
        figsize=figsize,
        annotate_titles=True,
    )
    print(f"Saved 4x4 grid to: {out_path}")


if __name__ == "__main__":
    main()

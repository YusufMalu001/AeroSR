"""Create leakage-free image-level train / val / test splits from data/original/."""

from __future__ import annotations

import random
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
ORIGINAL_DIR = DATA_DIR / "original"


def main(seed: int = 42, n_train: int = 160, n_val: int = 40) -> None:
    images = sorted([p.name for p in ORIGINAL_DIR.glob("*.jpg")])
    if not images:
        raise RuntimeError(f"No .jpg images found in {ORIGINAL_DIR}")

    rng = random.Random(seed)
    rng.shuffle(images)

    train_imgs = images[:n_train]
    val_imgs = images[n_train : n_train + n_val]
    test_imgs = images[n_train + n_val :]

    (DATA_DIR / "train_images.txt").write_text("\n".join(train_imgs) + "\n")
    (DATA_DIR / "val_images.txt").write_text("\n".join(val_imgs) + "\n")
    (DATA_DIR / "test_images.txt").write_text("\n".join(test_imgs) + "\n")
    print(f"Splits created (seed={seed}): train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")


if __name__ == "__main__":
    main()

"""CLI for creating reproducible, reusable training augmentation tensors."""
import argparse

from .fixed_data import export_fixed_data


def parser():
    p = argparse.ArgumentParser(description="Save a fixed augmented Food-101 training dataset")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", required=True, help="New or empty directory for tensor shards")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8, help="Mixup alpha; 0 disables Mixup")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha; 0 disables CutMix")
    p.add_argument("--random-erasing", type=float, default=0.25, help="Erasing probability; 0 disables it")
    p.add_argument("--crop-min-scale", type=float, default=0.08)
    p.add_argument("--views-per-image", type=int, default=1,
                   help="Number of independently seeded saved variants per training image")
    p.add_argument("--shard-size", type=int, default=64, help="Maximum tensors per shard")
    p.add_argument("--storage-dtype", choices=("float32", "float16"), default="float32",
                   help="float32 preserves augmentation values; float16 saves disk with rounding")
    p.add_argument("--threads", type=int, default=2, help="CPU threads used while exporting")
    return p


def main(args=None):
    if args is None or isinstance(args, (list, tuple)):
        args = parser().parse_args(args)
    return export_fixed_data(args)


if __name__ == "__main__":
    main()

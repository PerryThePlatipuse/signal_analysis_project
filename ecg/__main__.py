"""Command-line entry points for the ECG course project."""

import argparse
from pathlib import Path

from ecg.augment import NOISE_TYPES
from ecg.sources import DATASETS
from ecg.splits import PROTOCOLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download a public source or its repeat-record subset")
    download.add_argument("dataset", choices=DATASETS)
    download.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    download.add_argument("--workers", type=int, default=8)
    local = commands.add_parser("import-archive", help="import a downloaded source archive offline")
    local.add_argument("dataset", choices=DATASETS)
    local.add_argument("archive", type=Path)
    local.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    prepare = commands.add_parser("prepare", help="prepare whole records and inventory")
    prepare.add_argument("dataset", choices=DATASETS)
    prepare.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    prepare.add_argument("--out", type=Path)
    split = commands.add_parser("split", help="build known and unseen participant manifests")
    split.add_argument("prepared", type=Path)
    split.add_argument("--seed", type=int, default=42)
    for name in ("baseline", "train"):
        command = commands.add_parser(name)
        command.add_argument("prepared", type=Path)
        command.add_argument("--protocol", choices=PROTOCOLS, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "baseline":
            command.add_argument("--dimensions", type=int, default=16)
        else:
            command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
            command.add_argument("--epochs", type=int, default=20)
            command.add_argument("--batch-size", type=int, default=32)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--augment", action="store_true", help="add noise to training signals")
    evaluation = commands.add_parser("evaluate", help="score identification and verification on held-out ECGs")
    evaluation.add_argument("prepared", type=Path)
    evaluation.add_argument("--protocol", choices=PROTOCOLS, required=True)
    evaluation.add_argument("--method", choices=("pca", "cnn", "correlation", "dtw"), required=True)
    evaluation.add_argument("--model-run", type=Path)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--negatives", type=int, default=10)
    evaluation.add_argument("--max-participants", type=int, help="limit gallery size for exploratory runs")
    evaluation.add_argument("--query-noise", choices=NOISE_TYPES, help="perturb only held-out test queries")
    evaluation.add_argument("--noise-rms", type=float, help="noise RMS relative to each clean waveform")
    compare = commands.add_parser("compare", help="compute correlation and DTW for two original records")
    compare.add_argument("prepared", type=Path)
    compare.add_argument("left")
    compare.add_argument("right")
    args = parser.parse_args()

    if args.command == "download":
        from ecg.sources import download

        download(args.dataset, args.raw_root, args.workers)
    elif args.command == "import-archive":
        from ecg.sources import import_archive

        import_archive(args.dataset, args.archive, args.raw_root)
    elif args.command == "prepare":
        from ecg.preprocess import prepare

        prepare(args.dataset, args.raw_root / args.dataset,
                args.out or Path("data/prepared") / args.dataset)
    elif args.command == "split":
        from ecg.splits import write_splits

        write_splits(args.prepared, args.seed)
    elif args.command == "baseline":
        from ecg.classical import baseline

        baseline(args.prepared, args.protocol, args.out, args.dimensions)
    elif args.command == "evaluate":
        from ecg.evaluate import evaluate

        evaluate(args.prepared, args.protocol, args.method, args.out, args.model_run,
                 args.negatives, args.seed, args.max_participants, args.query_noise, args.noise_rms)
    elif args.command == "compare":
        from ecg.classical import compare

        compare(args.prepared, args.left, args.right)
    elif args.command == "train":
        from ecg.neural import train

        train(args.prepared, args.protocol, args.out, args.device, args.epochs, args.batch_size, args.seed, args.augment)


if __name__ == "__main__":
    main()

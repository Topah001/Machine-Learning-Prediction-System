import argparse

from .model import evaluate, predict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and use an Iris classifier.")
    subparsers = parser.add_subparsers(dest="command")
    prediction = subparsers.add_parser("predict", help="predict from four Iris measurements")
    prediction.add_argument("features", type=float, nargs=4, metavar="CM")
    parser.add_argument("--seed", type=int, default=42, help="reproducible split seed")
    parser.add_argument(
        "--test-size", type=float, default=0.25, help="held-out fraction for evaluation"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        try:
            label, confidence = predict(args.features, seed=args.seed)
        except ValueError as error:
            parser.error(str(error))
        print(f"Predicted species: {label}")
        print(f"Model confidence: {confidence:.3f}")
        return 0

    _, metrics = evaluate(seed=args.seed, test_size=args.test_size)
    print(f"Test samples: {metrics.test_samples}")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Macro precision: {metrics.macro_precision:.3f}")
    return 0

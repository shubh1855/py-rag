import argparse


def normalize_scores(scores: list[float]) -> None:
    if not scores:
        return

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        normalized_scores = [1.0] * len(scores)
    else:
        normalized_scores = [
            (score - min_score) / (max_score - min_score) for score in scores
        ]

    for score in normalized_scores:
        print(f"* {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize scores using min-max normalization",
    )
    normalize_parser.add_argument(
        "scores",
        nargs="*",
        type=float,
        help="Scores to normalize",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

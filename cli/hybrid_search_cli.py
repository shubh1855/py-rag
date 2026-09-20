import argparse

from inverted_index import load_movies
from lib.hybrid_search import HybridSearch


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

    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Run weighted hybrid Search",
    )
    weighted_parser.add_argument(
        "query",
        type=str,
        help="Search",
    )
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight given to BM25 scores",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Run RRF hybrid Search",
    )
    rrf_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF constant",
    )
    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores)
        case "weighted-search":
            documents = load_movies()

            hybrid = HybridSearch(documents)

            results = hybrid.weighted_search(
                args.query,
                args.alpha,
                args.limit,
            )

            for i, result in enumerate(results[: args.limit], start=1):
                document = result["document"]

                print(f"{i}. {document['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {document['description'][:100]}...")
        case "rrf-search":
            documents = load_movies()

            hybrid = HybridSearch(documents)

            results = hybrid.rrf_search(
                args.query,
                args.k,
                args.limit,
            )

            for i, result in enumerate(results[: args.limit], start=1):
                document = result["document"]

                print(f"{i}. {document['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {document['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

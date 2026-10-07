import argparse

from lib.hybrid_search import (
    normalize_scores,
    rrf_search_command,
    weighted_search_command,
)


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
        help="Run weighted hybrid search",
    )

    weighted_parser.add_argument(
        "query",
        type=str,
        help="Search query",
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
        help="Run RRF hybrid search",
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

    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=[
            "spell",
            "rewrite",
            "expand",
        ],
        help="Query enhancement method",
    )

    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual"],
        help="Reranking method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)

            for score in normalized_scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            result = weighted_search_command(
                args.query,
                args.alpha,
                args.limit,
            )

            for i, res in enumerate(
                result["results"],
                start=1,
            ):
                metadata = res["metadata"]

                print(f"{i}. {res['title']}")

                print(f"  Hybrid Score: {res['score']:.3f}")

                print(
                    f"  BM25: "
                    f"{metadata.get('bm25_score', 0.0):.3f}, "
                    f"Semantic: "
                    f"{metadata.get('semantic_score', 0.0):.3f}"
                )

                print(f"  {res['document']}...")

        case "rrf-search":
            result = rrf_search_command(
                args.query,
                args.k,
                args.enhance,
                args.rerank_method,
                args.limit,
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query "
                    f"({result['enhance_method']}): "
                    f"'{result['original_query']}' "
                    f"-> "
                    f"'{result['enhanced_query']}'\n"
                )

            if result["reranked"]:
                print(
                    f"Re-ranking top {args.limit} "
                    f"results using "
                    f"{result['rerank_method']} "
                    f"method...\n"
                )

            print(
                f"Reciprocal Rank Fusion Results "
                f"for '{result['query']}' "
                f"(k={result['k']}):"
            )

            for i, res in enumerate(
                result["results"],
                start=1,
            ):
                print(f"\n{i}. {res['title']}")

                individual_score = res.get("individual_score")

                if individual_score is not None:
                    print(f"   Re-rank Score: {individual_score:.3f}/10")

                metadata = res["metadata"]

                print(f"   RRF Score: {metadata.get('rrf_score', res['score']):.3f}")

                print(
                    f"   BM25 Rank: "
                    f"{metadata.get('bm25_rank')}, "
                    f"Semantic Rank: "
                    f"{metadata.get('semantic_rank')}"
                )

                print(f"   {res['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import TypedDict

from inverted_index import load_movies
from lib.hybrid_search import HybridSearch


class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]


class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit: int = args.limit

    if limit <= 0:
        parser.error("--limit must be greater than 0")

    dataset_path = (
        Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"
    )

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        golden_dataset: GoldenDataset = json.load(dataset_file)

    movies = load_movies()
    searcher = HybridSearch(movies)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        results = searcher.rrf_search(
            query=query,
            k=60,
            limit=limit,
        )

        retrieved_titles = [result["title"] for result in results]
        relevant_titles = set(relevant_docs)

        relevant_count = sum(title in relevant_titles for title in retrieved_titles)

        precision = relevant_count / len(retrieved_titles) if retrieved_titles else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
        print()


if __name__ == "__main__":
    main()

import argparse
import os

from dotenv import load_dotenv
from inverted_index import load_movies
from lib.hybrid_search import HybridSearch
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


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


def enhance_query_with_spell(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    enhanced_query = response.choices[0].message.content

    if not enhanced_query:
        return query

    return enhanced_query.strip()


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    rewritten_query = response.choices[0].message.content

    if not rewritten_query:
        return query

    return rewritten_query.strip()


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
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="query enhancement method",
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

            query = args.query

            if args.enhance == "spell":
                enhanced_query = enhance_query_with_spell(query)

                if enhanced_query != query:
                    print(
                        f"Enhanced query ({args.enhance}): "
                        f"'{query}' -> '{enhanced_query}'\n"
                    )

                query = enhanced_query

            elif args.enhance == "rewrite":
                enhanced_query = rewrite_query(query)

                if enhanced_query != query:
                    print(
                        f"Enhanced query ({args.enhance}): "
                        f"'{query}' -> '{enhanced_query}'\n"
                    )

                query = enhanced_query

            hybrid = HybridSearch(documents)

            results = hybrid.rrf_search(
                query,
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

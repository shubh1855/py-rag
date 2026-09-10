import argparse
import json

from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    subparsers.add_parser(
        "verify",
        help="Verify the semantic search model",
    )

    embed_parser = subparsers.add_parser(
        "embed_text",
        help="Generate an embedding for text",
    )
    embed_parser.add_argument(
        "text",
        type=str,
        help="Text to embed",
    )

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings",
        help="Build or load movie embeddings",
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate an embedding for a search query",
    )
    embed_query_parser.add_argument(
        "query",
        type=str,
        help="Search query to embed",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using semantic search",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search = SemanticSearch()

            with open("data/movies.json", "r") as file:
                data = json.load(file)

            documents = data["movies"]

            search.load_or_create_embeddings(documents)

            results = search.search(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description']}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

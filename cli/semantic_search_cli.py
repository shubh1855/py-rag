import argparse
import json

from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def chunk_command(text: str, chunk_size: int) -> None:
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]

    print(f"Chunking {len(text)} characters")

    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


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
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into fixed-size chunks",
    )
    chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk",
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk",
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
        case "chunk":
            chunk_command(args.text, args.chunk_size)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

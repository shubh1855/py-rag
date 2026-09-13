import argparse
import json
import re
from typing import overload

from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size.")

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)

        start += chunk_size - overlap

    print(f"Chunking {len(text)} characters")

    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def semantic_chunk_command(
    text: str,
    max_chunk_size: int,
    overlap: int,
) -> None:
    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be greater than 0.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= max_chunk_size:
        raise ValueError("Overlap must be less than max chunk size.")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [sentence for sentence in sentences if sentence]

    chunks = []

    start = 0
    while start < len(sentences):
        chunk = " ".join(sentences[start : start + max_chunk_size])
        chunks.append(chunk)

        start += max_chunk_size - overlap

    print(f"Semantically chunking {len(text)} characters")

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
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between the chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into semantic sentence-based chunks",
    )
    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk",
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks",
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
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(
                args.text,
                args.max_chunk_size,
                args.overlap,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

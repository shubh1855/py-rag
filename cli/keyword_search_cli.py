import argparse

from inverted_index import InvertedIndex, tokenize_text


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str) -> None:
    print("Searching for:", query)

    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found. Run the build command first.")

    query_tokens = tokenize_text(query)

    results = []

    for token in query_tokens:
        docs = index.get_documents(token)

        for doc_id in docs:
            if doc_id not in results:
                results.append(doc_id)

            if len(results) == 5:
                break

        if len(results) == 5:
            break

    for doc_id in results:
        movie = index.docmap[doc_id]
        print(f"{doc_id}. {movie['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using BM25",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    subparsers.add_parser(
        "build",
        help="Build the inverted index",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)

        case "build":
            build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

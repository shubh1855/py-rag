import argparse

from inverted_index import InvertedIndex


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


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
            print("Searching for:", args.query)

        case "build":
            build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

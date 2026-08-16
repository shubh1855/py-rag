import argparse
import json
import string

from nltk.stem import PortStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)

            with open("data/movies.json", "r") as file:
                data = json.load(file)

            translator = str.maketrans("", "", string.punctuation)

            with open("data/stopwords.txt", "r") as file:
                stopwords = file.read().splitlines()

            query = args.query.lower().translate(translator)
            query_tokens = query.split()

            stopwords = [word.lower().translate(translator) for word in stopwords]

            query = args.query.lower().translate(translator)
            query_tokens = [token for token in query.split() if token not in stopwords]

            results = []

            for movie in data["movies"]:
                title = movie["title"].lower().translate(translator)
                title_tokens = title.split()

                if any(
                    query_token in title_token
                    for query_token in query_tokens
                    for title_token in title_tokens
                ):
                    results.append(movie)

            results = results[:5]

            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

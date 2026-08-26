import argparse
import math

from inverted_index import BM25_K1, InvertedIndex, tokenize_term, tokenize_text


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


def calculate_idf(index: InvertedIndex, term: str) -> float:
    total_documents = len(index.docmap)
    document_frequency = len(index.get_documents(term))

    return math.log((total_documents + 1) / (document_frequency + 1))


def calculate_tf_idf(index: InvertedIndex, doc_id: int, term: str) -> float:
    tf = index.get_tf(doc_id, term)
    idf = calculate_idf(index, term)

    return tf * idf


def tf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found. Run the build  command first.")
        return

    token = tokenize_term(term)

    print(index.get_tf(doc_id, token))


def idf_command(term: str) -> None:
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found. Run the build command first.")
        return

    token = tokenize_term(term)

    idf = calculate_idf(index, token)

    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found. Run the build command first.")
        return

    token = tokenize_term(term)

    tf_idf = calculate_tf_idf(index, doc_id, token)

    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf_command(term: str):
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found. Run the build command")
        return 0.0

    token = tokenize_term(term)

    return index.get_bm25_idf(token)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found. Run the build command first.")
        return 0.0

    token = tokenize_term(term)

    return index.get_bm25_tf(doc_id, token, k1)


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

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate inverse document frequency"
    )
    idf_parser.add_argument("term", type=str, help="Term to Calculate IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 TF score for a given document ID and term",
    )
    bm25_tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID",
    )
    bm25_tf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 TF score for",
    )
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help="Tunable BM25 K1 parameter",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)

        case "build":
            build_command()

        case "tf":
            tf_command(args.doc_id, args.term)

        case "idf":
            idf_command(args.term)

        case "tfidf":
            tfidf_command(args.doc_id, args.term)

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(
                args.doc_id,
                args.term,
                args.k1,
            )

            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

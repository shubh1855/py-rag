import argparse
import os

from dotenv import load_dotenv
from inverted_index import load_movies
from lib.hybrid_search import HybridSearch
from lib.search_utils import SearchResult
from openai import OpenAI

MODEL = "openrouter/free"


def generate_answer(
    query: str,
    results: list[SearchResult],
) -> str:
    """Generate a context-aware answer using retrieved search results."""
    docs = "\n\n".join(
        f"Title: {result.get('title', 'Unknown title')}\n"
        f"Description: {result.get('document', '')}"
        for result in results
    )

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.
If the documents do not contain enough information, say so rather than inventing facts.

Query: {query}

Documents:
{docs}

Answer:"""

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=1000,
    )

    if not response.choices:
        raise RuntimeError("The LLM returned no response choices.")

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The LLM returned an empty answer.")

    return answer.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)",
    )
    rag_parser.add_argument(
        "query",
        type=str,
        help="Search query for RAG",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query: str = args.query

            movies = load_movies()
            searcher = HybridSearch(movies)

            results = searcher.rrf_search(
                query=query,
                k=60,
                limit=5,
            )

            print("Search Results:")
            for result in results:
                print(f"- {result.get('title', 'Unknown title')}")

            answer = generate_answer(query, results)

            print("\nRAG Response:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

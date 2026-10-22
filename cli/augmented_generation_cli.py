import argparse
import os

from dotenv import load_dotenv
from inverted_index import load_movies
from lib.hybrid_search import HybridSearch
from lib.search_utils import SearchResult
from openai import OpenAI

MODEL = "openrouter/free"


def get_llm_client() -> OpenAI:
    """Create an OpenRouter client using the configured API key."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def format_results(results: list[SearchResult]) -> str:
    """Format retrieved movies with their descriptions."""
    return "\n\n".join(
        f"Title: {result['title']}\nDescription: {result['document']}"
        for result in results
    )


def generate_answer(
    query: str,
    results: list[SearchResult],
) -> str:
    """Generate a context-aware answer from retrieved documents."""
    docs = format_results(results)

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.
If the documents do not contain enough information, say so rather than inventing facts.

Query: {query}

Documents:
{docs}

Answer:"""

    client = get_llm_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    if not response.choices:
        raise RuntimeError("The LLM returned no response choices.")

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The LLM returned an empty answer.")

    return answer.strip()


def generate_summary(
    query: str,
    results: list[SearchResult],
) -> str:
    """Synthesize information across multiple retrieved movies."""
    formatted_results = format_results(results)

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{formatted_results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    client = get_llm_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    if not response.choices:
        raise RuntimeError("The LLM returned no response choices.")

    summary = response.choices[0].message.content

    if not summary:
        raise RuntimeError("The LLM returned an empty summary.")

    return summary.strip()


def generate_cited_answer(
    query: str,
    results: list[SearchResult],
) -> str:
    """Generate an answer that cites the retrieved documents."""
    documents = "\n\n".join(
        f"[{index}] Title: {result['title']}\nDescription: {result['document']}"
        for index, result in enumerate(results, start=1)
    )

    if not documents:
        documents = "No documents were retrieved."

    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- Only cite source numbers that appear in the provided documents
- Ensure each citation supports the associated claim
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    client = get_llm_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
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

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize multiple search results",
    )
    summarize_parser.add_argument(
        "query",
        type=str,
        help="Query to search and summarize",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize (default: 5)",
    )

    citations_parser = subparsers.add_parser(
        "citations",
        help="Answer a query with source citations",
    )
    citations_parser.add_argument(
        "query",
        type=str,
        help="Query to answer with citations",
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use (default: 5)",
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
                print(f"- {result['title']}")

            answer = generate_answer(query, results)

            print("\nRAG Response:")
            print(answer)

        case "summarize":
            query: str = args.query
            limit: int = args.limit

            if limit <= 0:
                parser.error("--limit must be greater than 0")

            movies = load_movies()
            searcher = HybridSearch(movies)
            results = searcher.rrf_search(
                query=query,
                k=60,
                limit=limit,
            )

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            summary = generate_summary(query, results)

            print("\nLLM Summary:")
            print(summary)

        case "citations":
            query: str = args.query
            limit: int = args.limit

            if limit <= 0:
                parser.error("--limit must be greater than 0")

            movies = load_movies()
            searcher = HybridSearch(movies)
            results = searcher.rrf_search(
                query=query,
                k=60,
                limit=limit,
            )

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            answer = generate_cited_answer(query, results)

            print("\nLLM Answer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

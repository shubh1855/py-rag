import os
from time import sleep
from typing import Literal, Required

from dotenv import load_dotenv
from openai import OpenAI

from .search_utils import SearchResult


class RerankedSearchResult(SearchResult, total=False):
    individual_score: Required[float]


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openrouter/free"


def llm_rerank_individual(
    query: str,
    documents: list[SearchResult],
    limit: int = 5,
) -> list[RerankedSearchResult]:
    scored_docs: list[RerankedSearchResult] = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        score_text = (response.choices[0].message.content or "").strip()

        try:
            score = float(score_text)
        except ValueError:
            score = 0.0

        score = max(0.0, min(10.0, score))

        scored_docs.append(
            {
                **doc,
                "individual_score": score,
            }
        )

        sleep(3)

    scored_docs.sort(
        key=lambda x: x["individual_score"],
        reverse=True,
    )

    return scored_docs[:limit]


def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual"] = "individual",
    limit: int = 5,
) -> list[SearchResult] | list[RerankedSearchResult]:
    if method == "individual":
        return llm_rerank_individual(
            query,
            documents,
            limit,
        )

    return documents[:limit]

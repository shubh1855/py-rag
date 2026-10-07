import os
import re
from time import sleep
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

from .search_utils import SearchResult

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openrouter/free"


def extract_score(response_text: str) -> int | None:
    """Extract a score from 0 to 10 from an LLM response."""
    match = re.search(r"\b(10|[0-9])\b", response_text)

    if match is None:
        return None

    return int(match.group(1))


def score_document(query: str, doc: SearchResult) -> int:
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

    response_text = ""

    for _ in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        response_text = (response.choices[0].message.content or "").strip()

        score = extract_score(response_text)
        if score is not None:
            return score

        sleep(1)

    raise ValueError(
        f"LLM did not return a numeric score after 3 attempts: {response_text!r}"
    )


def llm_rerank_individual(
    query: str,
    documents: list[SearchResult],
    limit: int = 5,
) -> list[SearchResult]:
    scored_docs: list[SearchResult] = []

    for doc in documents:
        score = score_document(query, doc)

        scored_docs.append(
            {
                **doc,
                "individual_score": float(score),
            }
        )

        sleep(3)

    scored_docs.sort(
        key=lambda result: result.get("individual_score", 0.0),
        reverse=True,
    )

    return scored_docs[:limit]


def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual"] = "individual",
    limit: int = 5,
) -> list[SearchResult]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)

    return documents[:limit]

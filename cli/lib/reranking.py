import json
import os
import re
from time import sleep
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

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
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"


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
            messages=[{"role": "user", "content": prompt}],
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


def llm_rerank_batch(
    query: str,
    documents: list[SearchResult],
    limit: int = 5,
) -> list[SearchResult]:
    if not documents:
        return []

    doc_list_str = "\n".join(
        f"{doc['id']}: {doc['title']} - {doc['document']}" for doc in documents
    )

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = (response.choices[0].message.content or "").strip()

    try:
        ranked_ids = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON for batch reranking: {response_text!r}"
        ) from exc

    if not isinstance(ranked_ids, list) or not all(
        isinstance(movie_id, int) and not isinstance(movie_id, bool)
        for movie_id in ranked_ids
    ):
        raise ValueError("LLM batch ranking must be a JSON array of integers")

    document_map = {doc["id"]: doc for doc in documents}
    ranked_results: list[SearchResult] = []
    ranked_document_ids: set[int] = set()

    for rank, movie_id in enumerate(ranked_ids, start=1):
        doc = document_map.get(movie_id)

        if doc is None or movie_id in ranked_document_ids:
            continue

        ranked_results.append(
            {
                **doc,
                "metadata": {
                    **doc.get("metadata", {}),
                    "rerank_rank": rank,
                },
            }
        )
        ranked_document_ids.add(movie_id)

    remaining_rank = len(ranked_results) + 1

    for doc in documents:
        if doc["id"] in ranked_document_ids:
            continue

        ranked_results.append(
            {
                **doc,
                "metadata": {
                    **doc.get("metadata", {}),
                    "rerank_rank": remaining_rank,
                },
            }
        )
        ranked_document_ids.add(doc["id"])
        remaining_rank += 1

    ranked_results.sort(
        key=lambda result: result["metadata"].get("rerank_rank", float("inf"))
    )

    return ranked_results[:limit]


def llm_rerank_cross_encoder(
    query: str,
    documents: list[SearchResult],
    limit: int = 5,
) -> list[SearchResult]:
    if not documents:
        return []

    pairs: list[list[str]] = []

    for doc in documents:
        pairs.append(
            [
                query,
                f"{doc.get('title', '')} - {doc.get('document', '')}",
            ]
        )

    cross_encoder = CrossEncoder(
        CROSS_ENCODER_MODEL,
        device="cpu",
    )

    scores = cross_encoder.predict(pairs)

    scored_docs: list[SearchResult] = []

    for doc, score in zip(documents, scores):
        scored_docs.append(
            {
                **doc,
                "metadata": {
                    **doc.get("metadata", {}),
                    "cross_encoder_score": float(score),
                },
            }
        )

    scored_docs.sort(
        key=lambda result: result["metadata"].get("cross_encoder_score", float("-inf")),
        reverse=True,
    )

    return scored_docs[:limit]


def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual", "batch", "cross_encoder"] = "batch",
    limit: int = 5,
) -> list[SearchResult]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)

    if method == "batch":
        return llm_rerank_batch(query, documents, limit)

    return llm_rerank_cross_encoder(query, documents, limit)


def evaluate_results(
    query: str,
    documents: list[SearchResult],
) -> list[int]:
    """Evaluate search results on a relevance scale from 0 to 3."""
    if not documents:
        return []

    formatted_results = [
        f"{doc['id']}: {doc['title']} - {doc['document']}" for doc in documents
    ]

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents.
Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

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

    try:
        scores = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON for result evaluation: {response_text!r}"
        ) from exc

    if (
        not isinstance(scores, list)
        or len(scores) != len(documents)
        or not all(
            isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 3
            for score in scores
        )
    ):
        raise ValueError(
            "LLM evaluation must return one integer score from 0 to 3 "
            "for each search result"
        )

    return scores

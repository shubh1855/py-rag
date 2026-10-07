from typing import Any, NotRequired, TypedDict

SCORE_PRECISION = 4

DEFAULT_ALPHA = 0.5
DEFAULT_SEARCH_LIMIT = 5
RRF_K = 60
SEARCH_MULTIPLIER = 5


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]
    individual_score: NotRequired[float]


def format_search_result(
    doc_id: int,
    title: str,
    document: str,
    score: float,
    metadata: dict[str, Any] | None = None,
    **extra_metadata: Any,
) -> SearchResult:
    result_metadata = metadata.copy() if metadata else {}
    result_metadata.update(extra_metadata)

    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": result_metadata,
    }

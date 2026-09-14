SCORE_PRECISION = 4


def format_search_result(
    id,
    title,
    document,
    score,
    metadata=None,
):
    return {
        "id": id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }

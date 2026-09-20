import os

from inverted_index import InvertedIndex

from .semantic_search import ChunkedSemanticSearch


def rrf_score(rank: int, k: int) -> float:
    return 1 / (k + rank)


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: float = 0.5,
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()

        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self,
        query: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self,
        query: str,
        alpha: float,
        limit: int = 5,
    ) -> list[dict]:
        search_limit = limit * 500

        bm25_results = self._bm25_search(
            query,
            search_limit,
        )

        semantic_results = self.semantic_search.search_chunks(
            query,
            search_limit,
        )

        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [result["score"] for result in semantic_results]

        normalized_bm25 = normalize(bm25_scores)
        normalized_semantic = normalize(semantic_scores)

        movie_scores = {}

        for (doc_id, _), score in zip(
            bm25_results,
            normalized_bm25,
        ):
            movie_scores.setdefault(
                doc_id,
                {
                    "document": self.idx.docmap[doc_id],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                },
            )

            movie_scores[doc_id]["bm25_score"] = score

        for result, score in zip(
            semantic_results,
            normalized_semantic,
        ):
            doc_id = result["id"]

            movie_scores.setdefault(
                doc_id,
                {
                    "document": self.idx.docmap[doc_id],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                },
            )

            movie_scores[doc_id]["semantic_score"] = score

        results = []

        for doc_id, data in movie_scores.items():
            bm25_score = data["bm25_score"]
            semantic_score = data["semantic_score"]

            score = hybrid_score(
                bm25_score,
                semantic_score,
                alpha,
            )

            results.append(
                {
                    "id": doc_id,
                    "document": data["document"],
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                    "hybrid_score": score,
                }
            )

        results.sort(
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )
        return results

    def rrf_search(
        self,
        query: str,
        k: int,
        limit: int = 10,
    ) -> list[dict]:
        search_limit = limit * 500

        bm25_results = self._bm25_search(
            query,
            search_limit,
        )

        semantic_results = self.semantic_search.search_chunks(
            query,
            search_limit,
        )

        document_scores = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            if doc_id not in document_scores:
                document_scores[doc_id] = {
                    "document": self.idx.docmap[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }

            document_scores[doc_id]["bm25_rank"] = rank
            document_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]

            if doc_id not in document_scores:
                document_scores[doc_id] = {
                    "document": self.idx.docmap[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }

            document_scores[doc_id]["semantic_rank"] = rank
            document_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

        results = list(document_scores.values())

        results.sort(
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        return results

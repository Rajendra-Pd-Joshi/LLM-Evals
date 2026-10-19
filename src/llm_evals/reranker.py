import os
import requests
from langsmith import traceable
from src.llm_evals.retriever import load_store

RERANK_MODEL = "voyageai/rerank-2.5-lite"
RERANK_URL = "https://openrouter.ai/api/v1/rerank"


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_store()
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        candidates = self.store.similarity_search(query, k=self.fetch_k)
        try:
            resp = requests.post(
                RERANK_URL,
                headers=self.headers,
                json={
                    "model": RERANK_MODEL,
                    "query": query,
                    "documents": [d.page_content for d in candidates],
                    "top_n": self.top_k,
                },
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()["results"]  # [{"index": i, "relevance_score": s}, ...]
            return [candidates[r["index"]] for r in results]
        except Exception as e:
            print(f"Rerank failed, falling back to bi-encoder order: {e}")
            return candidates[: self.top_k]
import os
import requests

from dotenv import load_dotenv
from src.llm_evals.retriever import load_store
from langsmith import traceable

load_dotenv()

RERANK_MODEL = "cohere/rerank-4-fast"   # or "voyageai/rerank-2.5-lite" once you've checked the name
RERANK_URL = "https://openrouter.ai/api/v1/rerank"


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_store()
        self.fetch_k = fetch_k
        self.top_k = top_k

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        # 1. OVER-RETRIEVE
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. RERANK
        response = requests.post(
            RERANK_URL,
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": [doc.page_content for doc in candidates],
                "top_n": self.top_k,
            },
            timeout=60,
        )
        response.raise_for_status()
        results = response.json()["results"]

        # 3. MAP BACK
        return [candidates[r["index"]] for r in results]
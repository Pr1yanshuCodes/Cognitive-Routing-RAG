"""
Vector-based persona routing.

Each persona's `routing_text` was embedded once (see rag/ingest.py) into the
`persona_router` ChromaDB collection. Routing a query is a single similarity
search against those 4 vectors -- no LLM call, no extra latency, deterministic.
"""
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

import config
from personas.definitions import PERSONA_BY_ID, Persona


class PersonaRouter:
    def __init__(self, client: chromadb.ClientAPI | None = None, embedder=None):
        self.client = client or chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.embedder = embedder or HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    def route(self, query: str, top_k: int = None) -> list[tuple[Persona, float]]:
        """Returns [(persona, similarity_score), ...] sorted best-first.

        ChromaDB returns L2 distance by default; we convert to a rough
        similarity score (1 / (1 + distance)) purely for display -- ordering
        is what matters, and distance ordering already gives us that.
        """
        top_k = top_k or config.ROUTER_TOP_K
        try:
            collection = self.client.get_collection("persona_router")
        except Exception as e:
            raise RuntimeError(
                "persona_router collection not found. Run `python rag/ingest.py` first."
            ) from e

        query_embedding = self.embedder.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        persona_ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        routed = []
        for pid, dist in zip(persona_ids, distances):
            persona = PERSONA_BY_ID.get(pid)
            if persona:
                similarity = 1.0 / (1.0 + dist)
                routed.append((persona, round(similarity, 4)))
        return routed

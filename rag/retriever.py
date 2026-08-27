"""Top-k retrieval from a persona's ChromaDB collection."""
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

import config


class PersonaRetriever:
    def __init__(self, client: chromadb.ClientAPI | None = None, embedder=None):
        self.client = client or chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.embedder = embedder or HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    def retrieve(self, persona_id: str, query: str, k: int = None) -> list[dict]:
        """Returns [{text, source, distance}, ...] sorted by relevance."""
        k = k or config.RETRIEVAL_K
        try:
            collection = self.client.get_collection(f"persona_docs_{persona_id}")
        except Exception:
            return []  # no docs ingested for this persona yet

        query_embedding = self.embedder.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=k)

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {"text": doc, "source": meta.get("source", "unknown"), "distance": dist}
            for doc, meta, dist in zip(docs, metadatas, distances)
        ]

    def format_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "(no grounding documents retrieved)"
        return "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)

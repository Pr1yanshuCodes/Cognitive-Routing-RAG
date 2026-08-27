"""
Ingests data/sample_docs/<persona_id>/*.txt into a per-persona ChromaDB
collection named `persona_docs_<persona_id>`, and separately builds the
`persona_router` collection used by routing/router.py (one vector per
persona, built from its routing_text).

Run directly: `python rag/ingest.py`
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from personas.definitions import PERSONAS


def get_embedder():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def ingest_persona_docs(client: chromadb.ClientAPI, embedder) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")

    for persona in PERSONAS:
        persona_dir = os.path.join(base_dir, persona.id)
        if not os.path.isdir(persona_dir):
            print(f"[skip] no sample_docs folder for '{persona.id}'")
            continue

        collection = client.get_or_create_collection(f"persona_docs_{persona.id}")

        texts, ids, metadatas = [], [], []
        for fname in sorted(os.listdir(persona_dir)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(persona_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            chunks = splitter.split_text(raw)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                ids.append(f"{persona.id}-{fname}-{i}")
                metadatas.append({"source": fname, "persona": persona.id})

        if not texts:
            print(f"[skip] no .txt content found for '{persona.id}'")
            continue

        embeddings = embedder.embed_documents(texts)
        # upsert so re-running ingest.py is idempotent
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"[ok] ingested {len(texts)} chunks into persona_docs_{persona.id}")


def build_router_index(client: chromadb.ClientAPI, embedder) -> None:
    collection = client.get_or_create_collection("persona_router")
    texts = [p.routing_text for p in PERSONAS]
    ids = [p.id for p in PERSONAS]
    metadatas = [{"persona": p.id, "name": p.name} for p in PERSONAS]
    embeddings = embedder.embed_documents(texts)
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"[ok] built persona_router index with {len(ids)} personas")


def main():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    embedder = get_embedder()
    ingest_persona_docs(client, embedder)
    build_router_index(client, embedder)
    print("\nDone. Vector store is ready at:", config.CHROMA_DIR)


if __name__ == "__main__":
    main()

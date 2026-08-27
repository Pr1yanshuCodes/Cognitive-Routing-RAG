# Cognitive Routing RAG Engine

A multi-persona RAG system that vector-routes a user's query to the most
relevant "expert persona," retrieves grounded context per persona from
ChromaDB, and — for open-ended questions — runs a **LangGraph debate**
between two personas that argue, rebut, and get synthesized into a final
balanced answer. All user input passes through a **prompt-injection
defense layer** before it ever reaches the LLM.

## Architecture

```
                     ┌─────────────────────┐
   user query ──────▶│  Injection Defense    │──▶ blocked? ──▶ END (logged)
                     └──────────┬───────────┘
                                │ clean
                                ▼
                     ┌─────────────────────┐
                     │  Vector Persona Router │  (ChromaDB cosine search
                     └──────────┬───────────┘   over persona embeddings)
                                │ top-2 personas
                                ▼
              ┌─────────────────────────────────┐
              │   Persona A argument (RAG)        │
              │   Persona B argument (RAG)         │  parallel, each grounded
              └────────────────┬─────────────────┘   in its own Chroma collection
                                ▼
                     ┌─────────────────────┐
                     │   Rebuttal round        │  each persona reads the other's
                     └──────────┬───────────┘   argument and responds once
                                ▼
                     ┌─────────────────────┐
                     │   Judge / Synthesizer  │  merges both sides into one
                     └──────────┬───────────┘   grounded, cited answer
                                ▼
                              final answer
```

## Components

| Module | Responsibility |
|---|---|
| `personas/definitions.py` | 4 personas (Economist, Ethicist, Engineer, Skeptic): system prompt, routing description, doc collection |
| `security/injection_defense.py` | Pattern + heuristic risk scoring, input sanitization, delimiter hardening |
| `rag/ingest.py` | Chunks and embeds `data/sample_docs/<persona>/*.txt` into per-persona ChromaDB collections |
| `rag/retriever.py` | Top-k retrieval from a persona's collection |
| `routing/router.py` | Embeds persona descriptions once; routes each query via cosine similarity, no LLM call needed |
| `graph/debate_graph.py` | LangGraph `StateGraph` wiring the flow above, with a conditional edge on injection risk |
| `main.py` | CLI entry point |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python rag/ingest.py        # build the vector store from sample_docs
python main.py "Should companies be allowed to use AI to set individual prices for each customer?"
```

First run downloads the local embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~80MB, no API key needed) and creates a `chroma_db/` folder — everything else is local except the Groq LLM calls.

## Why Vector Routing

An LLM-based router costs a full inference call just to decide *who should answer*.
Here, each persona is described once by a short expertise paragraph + a handful of
example questions, embedded once at startup. Routing a new query is then a single
cosine-similarity lookup against 4 vectors — sub-millisecond, free, and deterministic
enough to unit test. The LLM is only ever called to *generate content*, not to route.

## Prompt injection defense

`security/injection_defense.py` runs before routing. It:
- Flags known override patterns ("ignore previous instructions", "you are now...", "reveal your system prompt", role-play jailbreak framings, suspicious base64/long-token blobs).
- Computes a 0–1 risk score from pattern hits + structural heuristics (excessive control tokens, delimiter injection attempts).
- Above the configured threshold, the graph short-circuits to a `blocked` node instead of ever building a prompt — the LLM never sees the payload.
- Below threshold, the query is still wrapped in explicit data delimiters (`<user_query> ... </user_query>`) with a hardened system instruction telling the model to treat that block as data, never as instructions.

This is a defense-in-depth layer for a demo project, not a production-grade guardrail — see "Limitations" below.

## Extending

- **Add a persona**: add an entry to `PERSONAS` in `personas/definitions.py`, drop some `.txt` files in `data/sample_docs/<id>/`, rerun `rag/ingest.py`.
- **Swap the LLM**: `config.py` centralizes model name / temperature; any LangChain chat model works, not just Groq.
- **More debate rounds**: `graph/debate_graph.py`'s `rebuttal_node` is a single pass — loop it N times by adding a conditional edge back to itself keyed on `state["round"]`.

## Limitations

- The injection-defense layer is regex/heuristic based, not a trained classifier — it stops common jailbreak phrasing, not novel obfuscated attacks.
- Vector routing picks personas by *topical* similarity, not by argument quality — a persona can be routed in and have little to actually say.
- No conversation memory across CLI invocations; each query is a fresh graph run.

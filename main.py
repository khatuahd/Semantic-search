# api/main.py
from typing import List
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mistralai import Mistral
from qdrant_client import QdrantClient

# -----------------------------
# CONFIG via variables d’environnement
# -----------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://bf703566-9c4d-4187-9d56-8b3f5bd05dbb.europe-west3-0.gcp.cloud.qdrant.io:6333",
)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_records")
MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set")
if not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_API_KEY is not set")

TOP_K = 5  # nombre de résultats retournés

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# CLIENTS
# -----------------------------
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
####
import inspect
import logging

logging.info("QDRANT CLIENT TYPE: %s", type(qdrant_client))
logging.info("QDRANT CLIENT MODULE: %s", type(qdrant_client).__module__)
logging.info("QDRANT CLIENT DIR HAS search_points? %s", hasattr(qdrant_client, "search_points"))
logging.info("QDRANT CLIENT DIR HAS search? %s", hasattr(qdrant_client, "search"))

####
# -----------------------------
# MODELES Pydantic
# -----------------------------
class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    id: str
    score: float  # similarité cosinus (Distance.COSINE)
    text: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


# -----------------------------
# APP FastAPI
# -----------------------------
app = FastAPI(title="Semantic Search API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    logger.info(f"Search query={query!r}, top_k={TOP_K}")

    # 1) Embedding Mistral
    try:
        emb_response = mistral_client.embeddings.create(
            model=MISTRAL_EMBED_MODEL,
            inputs=[query],
        )
        embedding = emb_response.data[0].embedding
    except Exception as e:
        logger.exception("Error while generating embeddings with Mistral")
        raise HTTPException(
            status_code=502,
            detail=f"Embedding generation failed: {e}",
        )

    # 2) Recherche vectorielle dans Qdrant
    try:
        hits = qdrant_client.search_points(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=TOP_K,
            with_payload=True,
        )
    except Exception as e:
        logger.exception("Error while searching in Qdrant")
        raise HTTPException(
            status_code=502,
            detail=f"Vector search failed: {e}",
        )

    # 3) Construction de la réponse
    results: List[SearchResult] = []
    for hit in hits:
        payload = hit.payload or {}
        text = (
            payload.get("text")
            or payload.get("summary")
            or payload.get("content")
            or ""
        )

        results.append(
            SearchResult(
                id=str(hit.id),
                score=float(hit.score),
                text=text,
            )
        )

    return SearchResponse(results=results)

# api/main.py
from typing import List
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mistralai import Mistral
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http import models as rest  # <- IMPORTANT (PayloadSchemaType)

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

# -----------------------------
# MODELES Pydantic
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    # patientId doit être fourni dans le JSON
    patient_id: str = Field(..., alias="patientId")

    class Config:
        populate_by_name = True  # accepte patient_id ou patientId


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    doc_id: str | None = None
    patient_id: str | None = None  # meta.patientId


class SearchResponse(BaseModel):
    results: List[SearchResult]


# -----------------------------
# APP FastAPI
# -----------------------------
app = FastAPI(title="Semantic Search API")


@app.on_event("startup")
def ensure_qdrant_patient_index():
    """
    Assure que l'index payload meta.patientId existe dans Qdrant.
    Nécessaire si Qdrant est en strict_mode (unindexed filtering interdit).
    Idempotent: si l'index existe déjà, ne fait rien.
    """
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        payload_schema = getattr(info, "payload_schema", None) or {}

        if "meta.patientId" in payload_schema:
            logger.info("Qdrant payload index already exists: meta.patientId")
            return

        logger.info("Creating Qdrant payload index: meta.patientId (keyword)")
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="meta.patientId",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        logger.info("Qdrant payload index created: meta.patientId")
    except Exception:
        # IMPORTANT: ne pas crasher le service si Qdrant a un souci temporaire au boot
        logger.exception("Failed to ensure Qdrant index meta.patientId at startup")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _build_patient_filter(patient_id: str) -> Filter:
    # Filtre strict sur meta.patientId
    return Filter(
        must=[
            FieldCondition(
                key="meta.patientId",
                match=MatchValue(value=patient_id),
            )
        ]
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    patient_id = request.patient_id.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if not patient_id:
        raise HTTPException(status_code=400, detail="patientId must not be empty.")

    logger.info("Search query=%r, top_k=%s, patientId=%r", query, TOP_K, patient_id)

    # 1) Embedding Mistral
    try:
        emb_response = mistral_client.embeddings.create(
            model=MISTRAL_EMBED_MODEL,
            inputs=[query],
        )
        embedding = emb_response.data[0].embedding
    except Exception as e:
        logger.exception("Error while generating embeddings with Mistral")
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {e}")

    # 2) Recherche vectorielle dans Qdrant + filtre patient obligatoire
    q_filter = _build_patient_filter(patient_id=patient_id)

    try:
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=TOP_K,
            with_payload=True,
            query_filter=q_filter,
        )
    except Exception as e:
        logger.exception("Error while searching in Qdrant")
        raise HTTPException(status_code=502, detail=f"Vector search failed: {e}")

    # 3) Construction de la réponse
    results: List[SearchResult] = []
    for hit in hits:
        payload = hit.payload or {}
        meta = payload.get("meta") or {}

        text = payload.get("text") or payload.get("summary") or payload.get("content") or ""

        results.append(
            SearchResult(
                id=str(hit.id),
                score=float(hit.score),
                text=text,
                doc_id=payload.get("doc_id"),
                patient_id=meta.get("patientId"),
            )
        )

    return SearchResponse(results=results)
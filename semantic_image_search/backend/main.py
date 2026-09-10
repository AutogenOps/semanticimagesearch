import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from semantic_image_search.backend.config import Config
from semantic_image_search.backend.query_translator import translate_query
from semantic_image_search.backend.ingestion import IndexService
from semantic_image_search.backend.retriever import ImageSearchService
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


# ------------------------------------------------------------------
# Lazy singletons (initialized in lifespan, used in endpoints)
# ------------------------------------------------------------------
search_service: Optional[ImageSearchService] = None
index_service: Optional[IndexService] = None


# ------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_service, index_service
    try:
        search_service = ImageSearchService()
        index_service = IndexService()
        log.info("Services initialized successfully")
    except Exception as e:
        log.error("Failed to initialize services on startup", error=str(e))
        # Don't raise — let endpoints handle failures gracefully
    yield
    # Shutdown cleanup (none needed currently)


# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------
app = FastAPI(
    title="Semantic Image Search API",
    description="CLIP + Qdrant + LLM Query Translator",
    version="1.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# CORS — allow all origins so the HTML frontend can call the API
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Static files — serve the HTML frontend
# ------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    log.info("Static files mounted", static_dir=str(STATIC_DIR))


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the HTML frontend at root."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Semantic Image Search API", "docs": "/docs"}


# ------------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "services_ready": search_service is not None}


# ------------------------------------------------------------------
# INGEST ENDPOINT
# ------------------------------------------------------------------
@app.post("/ingest")
def ingest_images(
    folder_path: Optional[str] = Query(None, description="Folder of images to index"),
):
    folder = folder_path or str(Config.IMAGES_ROOT)
    log.info("Ingest request received", folder=folder)

    if index_service is None:
        return JSONResponse(status_code=503, content={"error": "IndexService not initialized"})

    try:
        index_service.index_folder(folder)
        log.info("Ingestion completed", folder=folder)
        return {"message": f"Indexed images from {folder}"}

    except Exception as e:
        log.error("Ingestion failed", folder=folder, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ------------------------------------------------------------------
# TRANSLATE ENDPOINT
# ------------------------------------------------------------------
@app.get("/translate")
def translate(q: str):
    log.info("Translate request received", query=q)

    try:
        translated = translate_query(q)
        log.info("Query translated", original=q, translated=translated)
        return {"input": q, "translated": translated}

    except Exception as e:
        log.error("Translation failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ------------------------------------------------------------------
# TEXT SEARCH ENDPOINT
# ------------------------------------------------------------------
@app.get("/search-text")
def search_text_endpoint(
    q: str,
    k: int = 5,
    category: Optional[str] = None,
):
    log.info("Text search request received", query=q, top_k=k, category=category)

    if search_service is None:
        return JSONResponse(status_code=503, content={"error": "SearchService not initialized"})

    try:
        translated = translate_query(q)
        log.info("Query translated for text search", translated=translated)

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_text(translated, k=k, metadata_filter=metadata_filter)

        log.info("Text search completed", total_results=len(results.points))

        resp = [
            {
                "filename": p.payload.get("filename"),
                "path": p.payload.get("path"),
                "url": p.payload.get("url"),
                "category": p.payload.get("category"),
                "score": round(p.score, 4),
            }
            for p in results.points
        ]

        return {"query": q, "translated": translated, "k": k, "results": resp}

    except Exception as e:
        log.error("Text search failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ------------------------------------------------------------------
# IMAGE SEARCH ENDPOINT
# ------------------------------------------------------------------
@app.post("/search-image")
def search_image_endpoint(
    file: UploadFile = File(...),
    k: int = 5,
    category: Optional[str] = None,
):
    log.info("Image search request received", filename=file.filename)

    if search_service is None:
        return JSONResponse(status_code=503, content={"error": "SearchService not initialized"})

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "Only image files allowed"})

        Config.QUERY_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        query_path = Config.QUERY_IMAGE_ROOT / file.filename

        with query_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        log.info("Uploaded query image saved", path=str(query_path))

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_image(str(query_path), k=k, metadata_filter=metadata_filter)

        resp = [
            {
                "filename": p.payload.get("filename"),
                "path": p.payload.get("path"),
                "url": p.payload.get("url"),
                "category": p.payload.get("category"),
                "score": round(p.score, 4),
            }
            for p in results.points
        ]

        return {"query_image": file.filename, "k": k, "results": resp}

    except Exception as e:
        log.error("Image search failed", filename=file.filename, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})

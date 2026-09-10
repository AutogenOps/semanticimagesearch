"""
embeddings.py — CLIP embeddings via Hugging Face Inference API.

Replaces the previous torch/open_clip_torch dependency with lightweight
HTTP calls to the HF Inference API endpoint. This makes the service
deployable on Vercel (no 800MB torch bundle required).

Model: openai/clip-vit-base-patch32 → 512-dimensional embeddings.
"""
import base64
import time
from typing import List

import requests

from semantic_image_search.backend.config import Config
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


class HFCLIPEmbedder:
    """
    Calls the Hugging Face Inference API to produce CLIP embeddings
    for both text and images — no local GPU/torch required.
    """

    HF_API_BASE = "https://api-inference.huggingface.co/models"

    def __init__(self):
        if not Config.HF_API_KEY:
            raise SemanticImageSearchException(
                "HF_API_KEY is not set. "
                "Get a free key at https://huggingface.co/settings/tokens and add it to your environment."
            )

        self.model_url = f"{self.HF_API_BASE}/{Config.HF_CLIP_MODEL_ID}"
        self.headers = {
            "Authorization": f"Bearer {Config.HF_API_KEY}",
        }
        log.info(
            "HFCLIPEmbedder initialized",
            model=Config.HF_CLIP_MODEL_ID,
            url=self.model_url,
        )

    # ------------------------------------------------------------------
    # INTERNAL: POST with retry on model-loading (503)
    # ------------------------------------------------------------------
    def _post_with_retry(
        self,
        json_body: dict | None = None,
        data: bytes | None = None,
        content_type: str = "application/json",
        retries: int = 3,
    ) -> list:
        """
        Retry logic for HF Inference API cold starts:
        HF returns HTTP 503 with {"error": "Loading..."} when the model
        is warming up. We wait the suggested time and retry.
        """
        extra_headers = dict(self.headers)
        if content_type != "application/json":
            extra_headers["Content-Type"] = content_type

        for attempt in range(retries):
            if json_body is not None:
                resp = requests.post(self.model_url, headers=extra_headers, json=json_body, timeout=30)
            else:
                resp = requests.post(self.model_url, headers=extra_headers, data=data, timeout=30)

            if resp.status_code == 503:
                wait_time = int(resp.headers.get("X-Wait-For-Model", "20"))
                log.warning(
                    "HF model loading, retrying...",
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                )
                time.sleep(min(wait_time, 20))
                continue

            resp.raise_for_status()
            return resp.json()

        raise SemanticImageSearchException(
            f"HF Inference API unavailable after {retries} retries (model still loading)"
        )

    # ------------------------------------------------------------------
    # TEXT → VECTOR
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for embedding")

        log.info("Embedding text via HF API", text_preview=text[:60])

        try:
            result = self._post_with_retry(json_body={"inputs": text})

            # HF may return [[float,...]] or [float,...]
            vec = result[0] if isinstance(result[0], list) else result
            log.info("Text embedding successful", vector_dim=len(vec))
            return vec

        except Exception as e:
            log.error("HF text embedding failed", error=str(e))
            raise SemanticImageSearchException("Failed to embed text via HF API", e)

    # ------------------------------------------------------------------
    # IMAGE → VECTOR  (sends raw bytes)
    # ------------------------------------------------------------------
    def embed_image(self, image_path: str) -> List[float]:
        log.info("Embedding image via HF API", image=image_path)

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            result = self._post_with_retry(
                data=image_bytes,
                content_type="application/octet-stream",
            )

            vec = result[0] if isinstance(result[0], list) else result
            log.info("Image embedding successful", vector_dim=len(vec))
            return vec

        except SemanticImageSearchException:
            raise
        except Exception as e:
            log.error("HF image embedding failed", image=image_path, error=str(e))
            raise SemanticImageSearchException("Failed to embed image via HF API", e)

    # ------------------------------------------------------------------
    # BATCH IMAGE EMBEDDINGS
    # ------------------------------------------------------------------
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        log.info("Embedding batch images via HF API", total=len(image_paths))
        vectors = []
        for path in image_paths:
            vec = self.embed_image(path)
            vectors.append(vec)
        log.info("Batch embedding complete", total=len(vectors))
        return vectors


# ------------------------------------------------------------------
# LAZY SINGLETON
# ------------------------------------------------------------------
_embedder: HFCLIPEmbedder | None = None


def get_embedder() -> HFCLIPEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = HFCLIPEmbedder()
    return _embedder


# ------------------------------------------------------------------
# CONVENIENCE API (maintains backwards-compatible interface)
# ------------------------------------------------------------------
def embed_text(text: str) -> List[float]:
    return get_embedder().embed_text(text)


def embed_single_image(image_path: str) -> List[float]:
    return get_embedder().embed_image(image_path)


def embed_image_paths(image_paths: List[str]) -> List[List[float]]:
    return get_embedder().embed_images(image_paths)

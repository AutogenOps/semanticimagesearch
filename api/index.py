# Vercel Python serverless entry point.
# Vercel's @vercel/python runtime looks for an `app` object in api/index.py.
import sys
import os

# Ensure the project root is on sys.path so all package imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_image_search.backend.main import app  # noqa: F401 — re-exported for Vercel

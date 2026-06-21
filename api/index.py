"""
Vercel Serverless Entry Point.

Wraps the Flask app as a WSGI handler for Vercel's Python runtime.
Ensures the project root is on sys.path so that all relative imports
(models, utils, etc.) resolve correctly in the serverless environment.
"""
import os
import sys

# Add project root to sys.path so 'from models import ...' etc. work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app import app

# Vercel looks for a variable named `app` (WSGI) or `handler` (ASGI)
# Flask's app object is already a WSGI application, so this works directly.

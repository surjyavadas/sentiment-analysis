"""
Vercel Serverless Entry Point.

Wraps the Flask app as a WSGI handler for Vercel's Python runtime.
"""
from app import app

# Vercel looks for a variable named `app` (WSGI) or `handler` (ASGI)
# Flask's app object is already a WSGI application, so this works directly.

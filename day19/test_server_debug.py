#!/usr/bin/env python3
"""Debug script to test web server."""

import sys
import uvicorn
from src.web_server import create_app

if __name__ == "__main__":
    app = create_app("test_final")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="debug")
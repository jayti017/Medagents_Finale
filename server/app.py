"""
server/app.py — OpenEnv Entry Point
=====================================
Required by OpenEnv validator.
Must contain a main() function and if __name__ == '__main__' block.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main FastAPI app
from server import app

import uvicorn


def main():
    """
    Main entry point for OpenEnv validator.
    Starts the FastAPI server on port 7860.
    """
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
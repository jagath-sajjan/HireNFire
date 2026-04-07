"""
server/app.py — OpenEnv multi-mode deployment entry point
Exposes the HireNFire FastAPI+Gradio app for the `server` console script.

This file satisfies the openenv validate requirement:
  [project.scripts]
  server = "server.app:main"
"""

from __future__ import annotations
import os
import sys
import pathlib


def main() -> None:
    """Entry point called by the `serve` console script (pyproject.toml).

    Ensures the project root is on sys.path so that `import app` resolves
    whether we are running from source or from an installed package.
    """
    import uvicorn

    # Guarantee the project root (parent of this server/ package) is importable.
    # This is a no-op when running directly from source (root is already on
    # sys.path), but is essential when invoked via the installed `serve` script.
    _root = str(pathlib.Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    # Import the assembled FastAPI+Gradio ASGI app object directly.
    # Using the object (not a string) avoids the `uvicorn.run("app:app")`
    # string-import form which only works when CWD == project root.
    from app import app  # type: ignore[import]  # root-level module

    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    uvicorn.run(
        app,          # pass the object, not a string path
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code",
    )
    .pip_install(
        "pymupdf>=1.24.0",
        "sentence-transformers>=3.0.0",
        "rank-bm25>=0.2.2",
        "numpy>=2.0.0",
        "google-genai>=0.3.0",
        "python-dotenv>=1.0.0",
        "claude-agent-sdk>=0.1.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
    )
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=[
            ".env",
            ".env.*",
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            ".venv",
            ".git",
            "_diag",
            "baseline",
            "*.png",
            "uv.lock",
        ],
    )
)

app = modal.App(name="eyeonthemonster", image=image)

SECRETS = [modal.Secret.from_name("eyeonthemonster-secrets")]


@app.function(secrets=SECRETS, timeout=600)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root/app")
    from server import app as fastapi_app

    return fastapi_app

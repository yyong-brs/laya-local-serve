"""Air-gapped / locally-mounted checkpoint launcher for laya-serve.

The upstream ``laya-serve`` command builds its Router purely from environment
variables and always resolves checkpoints against the Hugging Face hub:
``LAYA_MODEL_PATH`` is honoured only by the one-shot quickstart
(``examples/docker/quickstart.py``), never by the HTTP server. So a stock
``laya-serve`` with a mounted checkpoint would still try to download weights.

This launcher instead builds ONE ``Agent`` from the directory named by
``LAYA_LOCAL_MODEL_PATH`` and attaches it to every canonical routing alias
(``english`` / ``multilingual`` / ``typed-decisions``), so every request --
explicit ``model`` or auto-routed -- hits the same resident checkpoint. No
network access is used.

It reuses the public ``laya.serve.create_app``, so the Jev-compatible
``POST /v1/systemone`` and ``GET /health`` surface is identical to the stock
server, and the stock healthcheck / auth / concurrency limits all apply.

Run by ``compose.local-serve.yaml``; the script is mounted into the image, so no
image rebuild beyond the standard CUDA build is required.
"""
import os

import uvicorn

from laya import load
from laya.router import Router
from laya.serve import create_app

# Canonical routing aliases the stock server understands. All of them point at
# the single local checkpoint, so a request naming any of them (or auto-routed)
# resolves without a Hugging Face download.
_ALIASES = ("multilingual", "english", "typed-decisions")


def main() -> None:
    device = os.environ.get("LAYA_DEVICE", "cuda")
    local = os.environ.get("LAYA_LOCAL_MODEL_PATH")
    if not local:
        raise SystemExit(
            "LAYA_LOCAL_MODEL_PATH is required: point it at the mounted checkpoint directory"
        )
    if not os.path.isdir(local):
        raise SystemExit("LAYA_LOCAL_MODEL_PATH is not a directory: %r" % local)

    # `default` is multilingual: it covers English text too, so an auto-routed
    # English request still resolves to the one loaded checkpoint.
    router = Router(device=device, default="multilingual")

    # Build the checkpoint once, on the selected device, then share that single
    # Agent instance across every alias -- this avoids loading the weights
    # once per alias (which would multiply VRAM usage).
    agent = load(local, device=device)
    for alias in _ALIASES:
        router.attach(alias, agent)

    app = create_app(router)
    uvicorn.run(
        app,
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=int(os.environ.get("LAYA_PORT", "8000")),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()

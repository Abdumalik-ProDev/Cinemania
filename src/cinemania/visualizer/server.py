"""Local HTTP server and static HTML exporter for the Neural Network Visualizer UI."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cinemania.config import (
    EXAMPLE_REVIEWS,
    REPORTS_DIR,
    RNN_ARTIFACTS_DIR,
    TRANSFORMER_CHECKPOINT_DIR,
    configure_logging,
)
from cinemania.visualizer.inspector import inspect_review_both

logger = logging.getLogger(__name__)

STATIC_HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"
DEFAULT_EXPORT_PATH = REPORTS_DIR / "nn_visualizer.html"

VISUALIZER_PRESETS: list[dict[str, str]] = [
    *EXAMPLE_REVIEWS,
    {
        "category": "strong-positive",
        "review": "An unforgettable masterpiece with brilliant acting and breathtaking cinematography.",
    },
    {
        "category": "strong-negative",
        "review": "A complete waste of time with wooden acting and a predictable, boring script.",
    },
]


def _render_html(
    *,
    live_mode: bool,
    presets: list[dict[str, str]],
    precomputed: dict[str, Any] | None = None,
) -> str:
    template = STATIC_HTML_PATH.read_text(encoding="utf-8")
    bootstrap = {
        "liveMode": live_mode,
        "presets": presets,
        "precomputed": precomputed or {},
    }
    safe_json = json.dumps(bootstrap).replace("</", "<\\/")
    script_tag = f"<script>window.__CINEMANIA_BOOTSTRAP__ = {safe_json};</script>"
    return template.replace("<!-- __CINEMANIA_BOOTSTRAP__ -->", script_tag)


def export_static_visualizer(
    output_path: Path | str = DEFAULT_EXPORT_PATH,
    *,
    presets: list[dict[str, str]] | None = None,
    rnn_dir: Path | str = RNN_ARTIFACTS_DIR,
    transformer_dir: Path | str = TRANSFORMER_CHECKPOINT_DIR,
) -> Path:
    """Export a standalone self-contained HTML visualizer with precomputed real model traces."""
    target_presets = presets if presets is not None else VISUALIZER_PRESETS
    precomputed: dict[str, Any] = {}
    for item in target_presets:
        review_text = item["review"]
        precomputed[review_text] = inspect_review_both(
            review_text,
            rnn_dir=rnn_dir,
            transformer_dir=transformer_dir,
        )

    html = _render_html(live_mode=False, presets=target_presets, precomputed=precomputed)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("Exported standalone Neural Network Visualizer to '%s'.", out)
    return out


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    rnn_dir: Path | str = RNN_ARTIFACTS_DIR,
    transformer_dir: Path | str = TRANSFORMER_CHECKPOINT_DIR,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer serving the interactive neural network UI."""
    # Precompute the default first preset so initial page load is instantaneous
    initial_review = VISUALIZER_PRESETS[1]["review"]
    initial_precomputed = {
        initial_review: inspect_review_both(
            initial_review,
            rnn_dir=rnn_dir,
            transformer_dir=transformer_dir,
        )
    }
    rendered_index = _render_html(
        live_mode=True,
        presets=VISUALIZER_PRESETS,
        precomputed=initial_precomputed,
    ).encode("utf-8")

    inspect_lock = threading.Lock()
    max_body_bytes = 65_536

    class VisualizerHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            if route in ("/", "/index.html"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(rendered_index)))
                self.end_headers()
                self.wfile.write(rendered_index)
                return
            if route == "/api/presets":
                self._send_json(HTTPStatus.OK, {"presets": VISUALIZER_PRESETS})
                return
            if route == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})

        def do_POST(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            if route != "/api/inspect":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})
                return
            try:
                raw_len = self.headers.get("Content-Length", "")
                try:
                    content_len = int(raw_len)
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header."})
                    return
                if content_len <= 0:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must not be empty."})
                    return
                if content_len > max_body_bytes:
                    self._send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"error": f"Request body exceeds {max_body_bytes} byte limit."},
                    )
                    return

                raw_body = self.rfile.read(content_len)
                try:
                    data = json.loads(raw_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Malformed UTF-8 JSON payload."})
                    return

                if not isinstance(data, dict):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON body must be an object."})
                    return

                review_text = data.get("review", "")
                if not isinstance(review_text, str) or not review_text.strip():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Review text must be a non-empty string."})
                    return

                with inspect_lock:
                    result = inspect_review_both(
                        review_text,
                        rnn_dir=rnn_dir,
                        transformer_dir=transformer_dir,
                    )
                self._send_json(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:  # pragma: no cover - defensive server boundary
                logger.exception("Failed to inspect review")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Internal inference error."})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.info("%s - %s", self.address_string(), format % args)

    return ThreadingHTTPServer((host, port), VisualizerHandler)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point to launch the live visualizer server or export standalone HTML."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Launch or export the Cinemania Neural Network Visualizer UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export a standalone self-contained HTML file (e.g., reports/nn_visualizer.html) and exit.",
    )
    args = parser.parse_args(argv)

    if args.export is not None:
        out = export_static_visualizer(args.export)
        print(f"Exported standalone visualizer to: {out}")
        return

    server = create_server(host=args.host, port=args.port)
    actual_port = server.server_address[1]
    print(f"Cinemania Neural Network Visualizer running at http://{args.host}:{actual_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down visualizer server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

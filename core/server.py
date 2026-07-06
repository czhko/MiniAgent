"""Backward-compat shim. Use `core.entry.server` directly."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.entry.server import ThreadingHTTPServer, AgentHTTPHandler  # noqa: E402

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="v0.4.3+ API Server")
    ap.add_argument("--port", type=int, default=18787)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), AgentHTTPHandler)
    url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"v0.4.3+ API: {url}/v1")
    print(f"Admin:   {url}/admin")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        srv.server_close()

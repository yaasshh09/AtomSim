"""Command-line entry point: `atomsim serve` launches the local app."""

import argparse
import threading
import webbrowser

import uvicorn

from atomsim.server.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomsim")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="launch the local server and open the app")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true")
    return parser


def _open_browser_soon(url: str) -> None:
    threading.Timer(1.5, webbrowser.open, args=(url,)).start()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        url = f"http://127.0.0.1:{args.port}"
        if not args.no_browser:
            _open_browser_soon(url)
        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":  # `python -m atomsim.cli`, alongside the console script
    main()

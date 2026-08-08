"""Command-line interface: vault <command>."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import db, permissions, reminders, summarize
from .actions import extract_actions
from .config import load_config, setup_logging
from .extractors import extract
from .ingest import ingest_paths
from .observability import init_observability
from .pii import scan as pii_scan
from .providers import PROVIDERS, ProviderError, UnavailableClient, build_client
from .agents import answer_question_agentic
from .qa import answer_question
from .vectors import VectorIndex


def _context(need_backend: bool = True):
    cfg = load_config()
    setup_logging(cfg)
    init_observability(cfg)
    conn = db.connect(cfg.db_path)
    try:
        client = build_client(cfg)
    except ProviderError as exc:
        if need_backend:
            sys.exit(f"error: {exc}")
        client = UnavailableClient(str(exc))
    if need_backend:
        if not client.is_up():
            hint = (
                f"start it with 'ollama serve' (url: {cfg.ollama_url})"
                if cfg.provider == "ollama"
                else "check its API key/network reachability"
            )
            sys.exit(f"error: {cfg.provider} backend is not reachable — {hint}")
        missing = client.missing_models()
        if missing:
            sys.exit(
                "error: required models not installed: "
                + ", ".join(missing)
                + "\ninstall with: "
                + " && ".join(f"ollama pull {m}" for m in missing)
            )
    return cfg, conn, client


def _read_input(args) -> str:
    if getattr(args, "file", None):
        return extract(Path(args.file).expanduser()).text
    if getattr(args, "text", None):
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    sys.exit("error: provide --file, --text, or pipe text on stdin")


def cmd_status(args) -> None:
    cfg, conn, client = _context(need_backend=False)
    up = client.is_up()
    embed_provider = cfg.embed_provider or cfg.provider
    provider_desc = cfg.provider if embed_provider == cfg.provider else f"{cfg.provider} (embed: {embed_provider})"
    print(f"data dir:   {cfg.data_dir}")
    print(f"provider:   {provider_desc}")
    print(f"backend:    {'up' if up else 'DOWN'}" + (f" ({cfg.ollama_url})" if cfg.provider == "ollama" else ""))
    if up:
        missing = client.missing_models()
        print(f"models:     {cfg.gen_model}, {cfg.embed_model}"
              + (f" — MISSING: {', '.join(missing)}" if missing else " — ok"))
    docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    errors = conn.execute("SELECT COUNT(*) AS n FROM documents WHERE status='error'").fetchone()["n"]
    print(f"documents:  {docs} ({errors} with errors), {chunks} chunks")


def cmd_ingest(args) -> None:
    cfg, conn, client = _context()
    paths = [Path(p) for p in args.paths] or cfg.watch_folders
    if not paths:
        sys.exit("error: no paths given and no watch_folders configured")
    report = ingest_paths(conn, client, paths, force=args.force)
    print(report.summary())
    for path, err in report.failed:
        print(f"  FAILED {path}: {err}", file=sys.stderr)


def cmd_watch(args) -> None:
    cfg, conn, client = _context()
    paths = [Path(p) for p in args.paths] or cfg.watch_folders
    if not paths:
        sys.exit("error: no paths given and no watch_folders configured")
    print(f"watching {', '.join(map(str, paths))} every {args.interval}s (ctrl-c to stop)")
    while True:
        report = ingest_paths(conn, client, paths)
        if report.added or report.updated or report.removed or report.failed:
            print(report.summary())
        time.sleep(args.interval)


def cmd_ask(args) -> None:
    cfg, conn, client = _context()
    index = VectorIndex(conn)

    # Determine which pipeline to use: CLI flag beats config.
    use_agentic = cfg.agentic_qa if args.agentic is None else args.agentic

    if use_agentic:
        result = answer_question_agentic(conn, index, client, args.question, cfg)

        # Pipeline / confidence header
        pipeline_label = f"[{result.pipeline} pipeline]"
        if result.verification is not None:
            conf_pct = int(result.verification.confidence * 100)
            tick = "✓" if result.verification.is_complete else "⚠"
            print(f"{pipeline_label} confidence: {conf_pct}% {tick}")
        elif result.pipeline != "fast":
            print(pipeline_label)

        # Gaps warning
        if result.verification and result.verification.gaps:
            gaps_str = ", ".join(result.verification.gaps)
            print(f"⚠  gaps: {gaps_str}")

        # Verbose: show how the question was decomposed
        if args.verbose and result.pipeline == "full" and len(result.sub_queries) > 1:
            print("Sub-queries:")
            for sq in result.sub_queries:
                print(f"  · {sq.question}")

        print()
        print(result.answer)

        if result.sources:
            print("\nSources:")
            for s in result.sources:
                loc = f" (page {s.page})" if s.page else ""
                print(f"  - {s.filename}{loc}")
    else:
        result = answer_question(conn, index, client, args.question, cfg)
        print(result.answer)
        if result.sources:
            print("\nSources:")
            for s in result.sources:
                loc = f" (page {s.page})" if s.page else ""
                print(f"  - {s.filename}{loc}")


def cmd_summarize(args) -> None:
    cfg, conn, client = _context()
    if args.doc:
        print(summarize.summarize_documents(conn, client, args.doc, args.mode))
        return
    if not args.paths:
        sys.exit("error: give file paths or --doc IDs")
    if len(args.paths) > 3:
        sys.exit("error: summarization takes at most 3 documents")
    texts = [extract(Path(p).expanduser()).text for p in args.paths]
    if len(texts) == 1:
        print(summarize.summarize_text(client, texts[0], args.mode))
    else:
        combined = "\n\n".join(
            f"## {Path(p).name}\n{t}" for p, t in zip(args.paths, texts)
        )
        print(summarize.summarize_text(client, combined, args.mode))


def cmd_actions(args) -> None:
    _, _, client = _context()
    items = extract_actions(client, _read_input(args))
    if args.json:
        print(json.dumps([i.to_dict() for i in items], indent=2))
        return
    if not items:
        print("no action items found")
        return
    for i in items:
        extras = [x for x in (i.owner and f"owner: {i.owner}", i.due_date and f"due: {i.due_date}") if x]
        suffix = f"  ({'; '.join(extras)})" if extras else ""
        print(f"[ ] {i.task}{suffix}")


def cmd_pii(args) -> None:
    _, _, client = _context(need_backend=not args.no_model)
    text = _read_input(args)
    spans = pii_scan(text, client=None if args.no_model else client, use_model=not args.no_model)
    if args.json:
        print(json.dumps([s.to_dict() for s in spans], indent=2))
        return
    if not spans:
        print("no PII found")
        return
    for s in spans:
        print(f"{s.kind:16} {s.start:>6}-{s.end:<6} {s.text!r}")


def cmd_remind(args) -> None:
    _, conn, client = _context(need_backend=False)
    llm = client if client.is_up() else None
    try:
        r = reminders.create_from_text(conn, args.text, client=llm)
    except reminders.ReminderParseError as exc:
        sys.exit(f"error: {exc}")
    print(f"reminder #{r.id}: {r.title} — due {r.due_at:%a %Y-%m-%d %H:%M}")


def cmd_reminders(args) -> None:
    _, conn, _ = _context(need_backend=False)
    if args.done:
        reminders.set_status(conn, args.done, "done")
        print(f"reminder #{args.done} marked done")
        return
    items = reminders.list_reminders(conn, include_done=args.all)
    if not items:
        print("no reminders")
        return
    for r in items:
        print(f"#{r.id:<4} [{r.status:8}] {r.due_at:%a %Y-%m-%d %H:%M}  {r.title}")


def cmd_models(args) -> None:
    cfg, conn, client = _context(need_backend=False)
    provider = args.provider or cfg.provider
    if args.provider and args.provider != cfg.provider:
        # Preview another provider without switching config.toml.
        from dataclasses import replace

        try:
            client = build_client(replace(cfg, provider=provider, embed_provider=provider))
        except ProviderError as exc:
            sys.exit(f"error: {exc}")
    try:
        available = client.list_models()
    except Exception as exc:  # noqa: BLE001 — surface any backend failure, not just ProviderError
        sys.exit(f"error: could not list models for provider '{provider}': {exc}")
    configured = {cfg.gen_model, cfg.embed_model} if provider == cfg.provider else set()

    def is_configured(name: str) -> bool:
        return any(name == c or name.split(":")[0] == c for c in configured)

    print(f"provider: {provider}"
          + (f"  (configured gen_model={cfg.gen_model}, embed_model={cfg.embed_model})" if configured else ""))
    if not available:
        print("no models reported by the backend")
        return
    for m in available:
        print(f"  {m}" + (" *" if is_configured(m) else ""))
    print("\nset gen_model / embed_model / provider in ~/.vault-assistant/config.toml to switch")


def cmd_docs(args) -> None:
    _, conn, _ = _context(need_backend=False)
    docs = db.list_documents(conn)
    if not docs:
        print("no documents ingested")
        return
    for d in docs:
        status = "" if d["status"] == "ok" else f"  ERROR: {d['error']}"
        print(f"#{d['id']:<4} {d['filename']}  ({d['chunk_count']} chunks){status}")


def cmd_folders(args) -> None:
    _, conn, _ = _context(need_backend=False)
    if args.remove is not None:
        try:
            permissions.remove_folder(conn, args.remove)
        except ValueError as exc:
            sys.exit(f"error: {exc}")
        print(f"removed folder permission #{args.remove}")
        return
    if args.set:
        path, level = args.set
        try:
            folder = permissions.set_folder(conn, path, level)
        except ValueError as exc:
            sys.exit(f"error: {exc}")
        print(f"#{folder.id:<4} {folder.path}  [{folder.access_level}]")
        return
    folders = permissions.list_folders(conn)
    if not folders:
        print("no folder permissions configured (all folders default to 'edit')")
        return
    for f in folders:
        prefix = f.path.rstrip("/") + "/"
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE path = ? OR path LIKE ?",
            (f.path, prefix + "%"),
        ).fetchone()["n"]
        print(f"#{f.id:<4} {f.path}  [{f.access_level}]  ({n} documents)")


def cmd_serve(args) -> None:
    from .api import run

    print(f"serving on http://127.0.0.1:{args.port}")
    run(port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vault", description="Offline personal document assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show system status").set_defaults(func=cmd_status)

    p = sub.add_parser("ingest", help="ingest files/folders into the index")
    p.add_argument("paths", nargs="*", help="files or folders (default: configured watch_folders)")
    p.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("watch", help="poll folders and ingest changes")
    p.add_argument("paths", nargs="*")
    p.add_argument("--interval", type=int, default=60, help="poll interval seconds")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("ask", help="ask a question about your documents")
    p.add_argument("question")
    agentic_group = p.add_mutually_exclusive_group()
    agentic_group.add_argument(
        "--agentic",
        dest="agentic",
        action="store_true",
        default=None,
        help="force the full agentic pipeline (decompose → retrieve → synthesize → verify)",
    )
    agentic_group.add_argument(
        "--no-agentic",
        dest="agentic",
        action="store_false",
        help="force the fast single-pass pipeline",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="show sub-queries and evidence details (full pipeline only)",
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("summarize", help="summarize 1-3 documents")
    p.add_argument("paths", nargs="*", help="file paths")
    p.add_argument("--doc", type=int, action="append", help="ingested document id (repeatable)")
    p.add_argument("--mode", choices=summarize.MODES, default="paragraph")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("actions", help="extract action items")
    p.add_argument("--file", help="input file")
    p.add_argument("--text", help="input text")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_actions)

    p = sub.add_parser("pii", help="flag PII in text or a file")
    p.add_argument("--file", help="input file")
    p.add_argument("--text", help="input text")
    p.add_argument("--no-model", action="store_true", help="regex pass only")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_pii)

    p = sub.add_parser("remind", help='create a reminder: vault remind "call Sam next Friday"')
    p.add_argument("text")
    p.set_defaults(func=cmd_remind)

    p = sub.add_parser("reminders", help="list reminders")
    p.add_argument("--all", action="store_true", help="include done")
    p.add_argument("--done", type=int, metavar="ID", help="mark reminder done")
    p.set_defaults(func=cmd_reminders)

    p = sub.add_parser("models", help="list available models for the configured (or a given) provider")
    p.add_argument(
        "--provider", choices=PROVIDERS, help="preview another provider's models without changing config.toml"
    )
    p.set_defaults(func=cmd_models)

    sub.add_parser("docs", help="list ingested documents").set_defaults(func=cmd_docs)

    p = sub.add_parser("folders", help="manage per-folder access control (readonly/edit/no_access)")
    p.add_argument(
        "--set", nargs=2, metavar=("PATH", "LEVEL"),
        help=f"set a folder's access level: {', '.join(permissions.LEVELS)}",
    )
    p.add_argument("--remove", type=int, metavar="ID", help="remove a folder's permission override")
    p.set_defaults(func=cmd_folders)

    p = sub.add_parser("serve", help="start the local web UI")
    p.add_argument("--port", type=int, default=8756)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Console entrypoint.

Usage::

    fastapi-assistant serve                     # HTTP API
    fastapi-assistant chat                      # interactive CLI (default)
    fastapi-assistant train                     # fine-tune the LoRA adapter
    fastapi-assistant eval --report r.json      # score the model on the eval set
    fastapi-assistant export merge --out DIR    # merge adapter into base weights
    fastapi-assistant export push ...           # publish a release to the HF Hub
    fastapi-assistant data validate             # check dataset quality/contamination
    fastapi-assistant data generate ...         # draft new Q&A pairs from the docs
    fastapi-assistant rag build                 # build the documentation index
    fastapi-assistant db migrate                # apply Postgres migrations
    fastapi-assistant keys create NAME          # issue an API key

The legacy flags ``--serve`` and ``--train`` still work. ``python -m
fastapi_assistant`` is equivalent to ``fastapi-assistant``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import get_settings
from .logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastapi-assistant",
        description="Fine-tuned small language model specialized in FastAPI.",
    )
    legacy = parser.add_mutually_exclusive_group()
    legacy.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    legacy.add_argument("--train", action="store_true", help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Run the HTTP API.")
    sub.add_parser("chat", help="Interactive terminal chat.")
    sub.add_parser("train", help="Fine-tune the LoRA adapter.")

    ev = sub.add_parser("eval", help="Evaluate the model on the held-out set.")
    ev.add_argument("--data", type=Path, help="Eval JSONL (default: EVAL set in DATA_DIR).")
    ev.add_argument("--limit", type=int, help="Only the first N items.")
    ev.add_argument("--max-new-tokens", type=int, default=256)
    ev.add_argument("--rag", action="store_true", help="Attach retrieved docs (RAG_INDEX_DIR).")
    ev.add_argument("--report", type=Path, help="Write the JSON report here.")
    ev.add_argument("--markdown", type=Path, help="Write a Markdown summary here.")
    ev.add_argument("--baseline", type=Path, help="Baseline report to gate against.")
    ev.add_argument("--max-regression", type=float, default=0.02)
    ev.add_argument("--min-score", type=float)
    ev.add_argument("--no-adapter", action="store_true", help="Score the base model only.")
    ev.add_argument(
        "--execute-code",
        action="store_true",
        help="Run generated code blocks in a subprocess (disposable environments only).",
    )

    ex = sub.add_parser("export", help="Merge and publish model releases.")
    ex_sub = ex.add_subparsers(dest="export_command", required=True)
    merge = ex_sub.add_parser("merge", help="Merge the LoRA adapter into the base weights.")
    merge.add_argument("--out", type=Path, required=True)
    push = ex_sub.add_parser("push", help="Upload a folder to a Hub repo and tag it.")
    push.add_argument("--folder", type=Path, required=True)
    push.add_argument("--repo", required=True)
    push.add_argument("--tag", required=True)
    push.add_argument("--kind", choices=["adapter", "gguf"], default="adapter")
    push.add_argument("--eval-report", type=Path)
    push.add_argument("--private", action="store_true")
    publish = ex_sub.add_parser(
        "publish-outputs",
        help="Publish a training run's adapter and GGUF and complete its model-release.json.",
    )
    publish.add_argument("out_dir", type=Path, help="The downloaded training output folder.")
    publish.add_argument("--namespace", required=True, help="Hugging Face user or organisation.")
    manifest = ex_sub.add_parser("env", help="Print env vars for a model-release.json.")
    manifest.add_argument("manifest", type=Path)
    manifest.add_argument("--format", choices=["dotenv", "yaml", "gcloud"], default="dotenv")

    data = sub.add_parser("data", help="Dataset tooling.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    validate = data_sub.add_parser("validate", help="Schema, code, duplicate, and leakage checks.")
    validate.add_argument("--train", type=Path)
    validate.add_argument("--eval", type=Path)
    validate.add_argument("--near-dup-threshold", type=float, default=0.7)
    data_sub.add_parser("stats", help="Print dataset statistics.")
    merge_data = data_sub.add_parser(
        "merge", help="Merge JSONL shards into the dataset (deduplicated)."
    )
    merge_data.add_argument("shards", nargs="+", type=Path)
    merge_data.add_argument("--split", choices=["train", "eval"], required=True)
    gen = data_sub.add_parser("generate", help="Draft Q&A pairs from the FastAPI docs with Claude.")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--pages", type=int, default=20, help="How many doc sections to sample.")
    gen.add_argument("--per-chunk", type=int, default=3)
    gen.add_argument("--ref", default=None, help="FastAPI git tag to read the docs from.")
    gen.add_argument("--model", default=None, help="Claude model id.")
    gen.add_argument("--seed", type=int, default=0)

    rag = sub.add_parser("rag", help="Retrieval index tooling.")
    rag_sub = rag.add_subparsers(dest="rag_command", required=True)
    build = rag_sub.add_parser("build", help="Build the documentation index.")
    build.add_argument("--target", choices=["local", "pgvector"], default=None)
    build.add_argument("--ref", default=None, help="FastAPI git tag to index.")
    build.add_argument("--workdir", type=Path, default=Path(".cache/fastapi-docs"))
    query = rag_sub.add_parser("query", help="Show retrieval results for a question.")
    query.add_argument("question")

    db = sub.add_parser("db", help="Database tooling.")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="Apply pending migrations.")

    keys = sub.add_parser("keys", help="API key management.")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    create = keys_sub.add_parser("create", help="Issue a key (printed once).")
    create.add_argument("name")
    keys_sub.add_parser("list", help="List keys.")
    revoke = keys_sub.add_parser("revoke", help="Revoke keys by display prefix.")
    revoke.add_argument("prefix")
    return parser


# -- command implementations --------------------------------------------------------
def _serve(settings) -> None:
    import uvicorn

    uvicorn.run(
        "fastapi_assistant.api:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_graceful_shutdown=30,
    )


def _eval(args, settings) -> int:
    from .engines import create_engine
    from .evaluation import gate, load_eval_items, render_markdown, run_eval

    if args.no_adapter:
        settings = settings.model_copy(update={"use_adapter": False, "finetuned_weights": False})
    items = load_eval_items(args.data or settings.eval_path, args.limit)
    engine = create_engine(settings)

    def progress(i, total, result):
        print(
            f"[{i}/{total}] {result.id}: score={result.score:.3f} ({result.latency_s:.1f}s)",
            file=sys.stderr,
        )

    retriever = None
    if args.rag:
        from .retrieval import create_retriever

        retriever = create_retriever(settings.model_copy(update={"rag_enabled": True}))
        assert retriever is not None  # rag_enabled=True always builds one
    report = run_eval(
        engine,
        items,
        max_new_tokens=args.max_new_tokens,
        execute_code=args.execute_code,
        meta={
            "model_version": settings.model_version,
            "data": str(args.data or settings.eval_path),
        },
        progress=progress,
        retriever=retriever,
        rag_top_k=settings.rag_top_k,
        rag_max_chars=settings.rag_max_context_chars,
    )
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(report, baseline)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown)
    result = gate(report, baseline, max_regression=args.max_regression, min_score=args.min_score)
    if not result.passed:
        for reason in result.reasons:
            print(f"GATE FAILED: {reason}", file=sys.stderr)
        return 1
    return 0


def _export(args, settings) -> int:
    from . import export

    if args.export_command == "merge":
        export.merge_adapter(settings, args.out)
    elif args.export_command == "push":
        report = (
            json.loads(args.eval_report.read_text(encoding="utf-8")) if args.eval_report else None
        )
        export.write_model_card(
            args.folder, settings=settings, version=args.tag, kind=args.kind, eval_report=report
        )
        sha = export.push_folder(args.folder, args.repo, tag=args.tag, private=args.private)
        print(sha)
    elif args.export_command == "publish-outputs":
        published = export.publish_training_outputs(
            args.out_dir, args.namespace, token=os.environ.get("HF_TOKEN")
        )
        target = args.out_dir / "model-release.json"
        target.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(published, indent=2))
    elif args.export_command == "env":
        env = export.manifest_env(export.load_manifest(args.manifest))
        if args.format == "yaml":
            for k, v in env.items():
                print(f'{k}: "{v}"')
        elif args.format == "gcloud":
            print(",".join(f"{k}={v}" for k, v in env.items()))
        else:
            for k, v in env.items():
                print(f"{k}={v}")
    return 0


def _data(args, settings) -> int:
    from . import datasets_tools as tools

    if args.data_command == "validate":
        report = tools.validate(
            args.train or settings.train_path,
            args.eval or settings.eval_path,
            near_dup_threshold=args.near_dup_threshold,
        )
        print(report.render())
        return 0 if report.ok else 1
    if args.data_command == "stats":
        print(json.dumps(tools.stats(settings.train_path, settings.eval_path), indent=2))
        return 0
    if args.data_command == "merge":
        target = settings.train_path if args.split == "train" else settings.eval_path
        added, skipped = tools.merge_shards(target, args.shards, split=args.split)
        print(f"Added {added} rows to {target} ({skipped} duplicates skipped)")
        return 0
    if args.data_command == "generate":
        from .datagen import generate_dataset

        n = generate_dataset(
            out_path=args.out,
            n_chunks=args.pages,
            per_chunk=args.per_chunk,
            ref=args.ref,
            model=args.model,
            seed=args.seed,
        )
        print(f"Wrote {n} draft rows to {args.out}. Review them before merging.")
        return 0
    return 2


def _rag(args, settings) -> int:
    if args.rag_command == "build":
        from .docs_corpus import (
            DEFAULT_FASTAPI_REF,
            chunk_pages,
            download_fastapi_repo,
            load_doc_pages,
        )
        from .retrieval import FastEmbedEmbedder, build_local_index, build_pgvector_index

        ref = args.ref or DEFAULT_FASTAPI_REF
        target = args.target or settings.rag_backend
        repo = download_fastapi_repo(ref, args.workdir)
        chunks = chunk_pages(load_doc_pages(repo))
        print(f"{len(chunks)} chunks from FastAPI {ref}")
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
        if target == "pgvector":
            if not settings.database_url:
                print("DATABASE_URL is required for --target pgvector", file=sys.stderr)
                return 2
            n = build_pgvector_index(
                chunks, embedder, settings.database_url, settings.rag_index_version
            )
            print(f"Indexed {n} chunks into pgvector (version {settings.rag_index_version})")
        else:
            out = build_local_index(
                chunks,
                embedder,
                settings.rag_index_dir,
                {"fastapi_ref": ref, "embedding_model": settings.embedding_model},
            )
            print(f"Wrote local index to {out}")
        return 0
    if args.rag_command == "query":
        from .retrieval import create_retriever

        retriever = create_retriever(settings.model_copy(update={"rag_enabled": True}))
        assert retriever is not None  # rag_enabled=True always builds one
        for p in retriever.retrieve(args.question, settings.rag_top_k):
            print(f"{p.score:.3f}  {p.heading}  {p.url}")
        return 0
    return 2


def _store(settings):
    from .storage import create_store

    store = create_store(settings)
    if store is None:
        print("DATABASE_URL is not set", file=sys.stderr)
        sys.exit(2)
    return store


def _db(args, settings) -> int:
    store = _store(settings)
    applied = store.migrate()
    print("Applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
    store.close()
    return 0


def _keys(args, settings) -> int:
    store = _store(settings)
    try:
        if args.keys_command == "create":
            key, record = store.create_key(args.name)
            print(f"Created key {record.prefix}... for {record.name!r} (id {record.id}).")
            print("Store it now; it cannot be shown again:")
            print(key)
        elif args.keys_command == "list":
            for r in store.list_keys():
                print(f"{r.id:>5}  {r.prefix}...  {'active ' if r.active else 'revoked'}  {r.name}")
        elif args.keys_command == "revoke":
            print(f"Revoked {store.revoke_key(args.prefix)} key(s)")
    finally:
        store.close()
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    command = args.command or ("serve" if args.serve else "train" if args.train else "chat")
    if command == "serve":
        _serve(settings)
    elif command == "train":
        from .training import train

        train(settings)
    elif command == "chat":
        from .cli import interactive_chat

        interactive_chat(settings=settings)
    else:
        handler = {
            "eval": _eval,
            "export": _export,
            "data": _data,
            "rag": _rag,
            "db": _db,
            "keys": _keys,
        }[command]
        sys.exit(handler(args, settings))


if __name__ == "__main__":
    main()

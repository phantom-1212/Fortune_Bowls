from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import read_text_from_path
from .pdf_loader import pdf_to_text
from .rag import ingest_document, retrieve, synthesize_answer


def main():
    parser = argparse.ArgumentParser(description="KB RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest files")
    p_ingest.add_argument("paths", nargs="+", help="Paths to files (txt/pdf)")

    p_query = sub.add_parser("query", help="Query")
    p_query.add_argument("question", help="User question")
    p_query.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "ingest":
        for p in args.paths:
            path = Path(p)
            text = pdf_to_text(str(path)) if path.suffix.lower() == ".pdf" else read_text_from_path(str(path))
            n = ingest_document(doc_id=path.name, source_path=str(path), text=text)
            print(f"Ingested {path} -> {n} chunks")
    elif args.cmd == "query":
        ctx = retrieve(args.question, top_k=args.top_k)
        ans = synthesize_answer(args.question, ctx)
        print(ans)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

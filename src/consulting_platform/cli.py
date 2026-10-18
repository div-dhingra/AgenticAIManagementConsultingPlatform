"""CLI for direct agent testing before the API/frontend exists."""

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from uuid import uuid4

from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import (
    ConsultContext,
    ConsultRequest,
    ServiceUnavailableError,
    UpstreamModelError,
    UpstreamTimeoutError,
)


def load_factory(spec: str):
    """Local developer-controlled integration hook, never exposed to customers."""
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)()


def main():
    parser = argparse.ArgumentParser(
        description="Run the consulting agents without an HTTP server."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("ask")
    run.add_argument("--question")
    run.add_argument("--request-json", type=Path)
    run.add_argument("--context-json", default="{}")
    run.add_argument("--mode", choices=["live", "demo"])
    knowledge_source = run.add_mutually_exclusive_group()
    knowledge_source.add_argument(
        "--knowledge-factory", help="Trusted local module:factory from the KB teammate"
    )
    knowledge_source.add_argument(
        "--evidence-file",
        type=Path,
        help="Explicit real-passage JSON for live-model tests without the full KB",
    )
    graph = sub.add_parser("graphs")
    graph.add_argument("--output-dir", type=Path, default=Path("docs/graphs"))
    args = parser.parse_args()
    if args.command == "graphs":
        service = create_consulting_service(settings=Settings(mode="demo"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        graphs = {
            "manager": service.graph,
            **{d.value: a.graph for d, a in service.specialists.items()},
        }
        for name, compiled in graphs.items():
            (args.output_dir / f"{name}.mmd").write_text(compiled.get_graph().draw_mermaid())
        print(f"Wrote {len(graphs)} Mermaid graphs to {args.output_dir}.")
        return
    settings = Settings.from_env()
    if args.mode:
        settings.mode = args.mode
    if args.evidence_file and settings.mode != "live":
        parser.error(
            "--evidence-file requires --mode live; it uses real model calls and real passages."
        )
    if args.request_json:
        request = ConsultRequest.model_validate_json(args.request_json.read_text())
    elif args.question:
        request = ConsultRequest(
            question=args.question, context=ConsultContext.model_validate_json(args.context_json)
        )
    else:
        parser.error("Supply --question or --request-json.")
    try:
        knowledge = load_factory(args.knowledge_factory) if args.knowledge_factory else None
        if args.evidence_file:
            from consulting_platform.agents.controlled_evidence import ControlledEvidenceService

            knowledge = ControlledEvidenceService.from_file(args.evidence_file)
        service = create_consulting_service(knowledge, settings=settings)
        result = asyncio.run(service.run_consultation(request, request_id=str(uuid4())))
    except (ServiceUnavailableError, UpstreamTimeoutError, UpstreamModelError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

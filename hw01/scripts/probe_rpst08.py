"""Offline CPU smoke probe for the separately provisioned RPST evaluator.

Run with .venv-train/Scripts/python.exe -B -X utf8 scripts/probe_rpst08.py.
The upstream score combines metre and rhyme heuristics; it is not a calibrated
quality probability. Failures and unrecognized analyses remain null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts/evaluators/rpst"
SMOKE_CASES = [
    {"id": "aa_pushkin", "text": "Мороз и солнце; день чудесный!\nЕще ты дремлешь, друг прелестный"},
    {"id": "nonrhyming", "text": "Я вижу за окошком дом.\nНа стол поставили тарелку."},
    {"id": "identical_endings", "text": "Мороз и солнце; день чудесный!\nМой друг сегодня так чудесный!"},
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_verified_tool(artifacts: Path):
    """Verify provisioned bytes before importing code or unpickling models."""
    version = json.loads((artifacts / "version.json").read_text(encoding="utf-8"))
    for item in version["verified_files"]:
        path = (artifacts / item["path"]).resolve()
        if not path.is_relative_to(artifacts.resolve()):
            raise ValueError("Manifest path escapes evaluator directory")
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Evaluator file changed: {item['path']}")

    # These settings affect this probe process only, never the training env.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(artifacts / p) for p in version["import_paths"]]
    import torch
    import russian_scansion

    torch.set_num_threads(2)
    with torch.inference_mode():
        tool = russian_scansion.create_rpst_instance(models_dir=str(artifacts / "models"))
    devices = {str(parameter.device) for parameter in tool.accentuator.stress_model.model.parameters()}
    if devices != {"cpu"}:
        raise RuntimeError(f"Unexpected evaluator devices: {devices}")
    return tool, torch, version


def analyze(tool, case: dict) -> dict:
    """Keep unknown/error distinct from a successfully computed zero score."""
    started = time.perf_counter()
    result = dict(case, status="unknown", score=None, meter=None, rhyme_scheme=None,
                  stressed_text=None, repeating=None, poor_poetry=None, diagnostic=None)
    try:
        alignment = tool.align(case["text"].splitlines())
        error = getattr(alignment, "error_text", None)
        if error or not alignment.meter or alignment.rhyme_scheme is None:
            result["diagnostic"] = error or "RPST did not determine metre/rhyme scheme"
        else:
            score = float(alignment.score)
            if not math.isfinite(score):
                raise ValueError("Non-finite RPST score")
            result.update(status="ok", score=score, meter=alignment.meter,
                          rhyme_scheme=alignment.rhyme_scheme,
                          stressed_text=alignment.get_stressed_lines(show_secondary_accentuation=True),
                          repeating=bool(tool.detect_repeating(alignment)),
                          poor_poetry=bool(tool.detect_poor_poetry(alignment)))
    except Exception as exc:
        result["status"] = "error"
        result["diagnostic"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--text-file", type=Path, help="Optional UTF-8 poem instead of the three smoke cases")
    parser.add_argument("--output", type=Path, help="Save JSON; otherwise print only")
    args = parser.parse_args()
    started = time.perf_counter()
    report = {"device": "cpu", "calibrated": False, "training": False,
              "score_semantics": "upstream combined metre/rhyme heuristic; not semantic quality",
              "results": None, "diagnostic": None}
    try:
        tool, torch, version = load_verified_tool(args.artifacts)
        report["revision"] = version["rpst_revision"]
        report["torch_version"] = torch.__version__
        report["load_seconds"] = time.perf_counter() - started
        cases = SMOKE_CASES if args.text_file is None else [
            {"id": args.text_file.stem, "text": args.text_file.read_text(encoding="utf-8")}
        ]
        with torch.inference_mode():
            report["results"] = [analyze(tool, case) for case in cases]
        report["status"] = "complete" if all(r["status"] == "ok" for r in report["results"]) else "partial"
    except Exception as exc:
        report.update(status="error", diagnostic=f"{type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = time.perf_counter() - started
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

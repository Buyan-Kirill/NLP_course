"""Replay one documented stage. Listing and checking never train a model.

Numerical implementations remain in the original, hash-bound trainers. This
module only names stages, binds their parents and keeps new runs separate from
the research history. New checkpoint hashes are expected after a fresh replay.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("runs/reproduction")
STAGES = {
    "pretrain": {
        "description": "D: random initialization, 630193557 prose targets",
        "source": "configs/pretrain.json",
        "script": "scripts/train_pretrain08.py", "budget": 630193557,
        "parent": None,
    },
    "pretrain_finish": {
        "description": "Continue D for 50M targets, inherited schedule with LR x0.5",
        "source": "configs/pretrain_finish.json",
        "script": "scripts/train_targeted10.py", "budget": None,
        "parent": "pretrain",
    },
    "poetry": {
        "description": "Pure poetry, first 1615779 targets; fresh Muon + AdamW",
        "source": "configs/poetry.json",
        "script": "scripts/train_poetry08.py", "budget": 1615779,
        "parent": "pretrain_finish",
    },
    "poetry_small_batch": {
        "description": "Continue the same poetry schedule to 3231558 targets, batch 4096",
        "source": "configs/poetry_small_batch.json",
        "script": "scripts/train_poetry16.py", "budget": 3231558,
        "parent": "poetry",
    },
    "ending_plan": {
        "description": "Planned400k: 50% ordinary poetry, 50% conditioned line targets",
        "source": "configs/ending_plan.json",
        "script": "scripts/train_poetry_units16.py", "budget": 400000,
        "parent": "poetry_small_batch",
    },

}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def recipe(stage, device="cuda", resume=False):
    """Bind the numerical recipe to replay parents and a portable runtime allowance."""
    spec = STAGES[stage]
    cfg = deepcopy(read(ROOT / spec["source"]))
    parent = None if spec["parent"] is None else RUNS / spec["parent"] / "last.pt"
    if parent is not None:
        if stage == "ending_plan":
            cfg["training"]["parent"] = parent.as_posix()
            cfg["parent_sha256"] = digest(ROOT / parent) if (ROOT / parent).exists() else "<run previous stage first>"
        else:
            cfg["parent"] = parent.as_posix()
    run_dir = RUNS / stage
    training = cfg.get("training", cfg)
    training["max_active_seconds"] = 7 * 24 * 3600
    cfg_path = Path("reproduction/generated") / (stage + ".json")
    command = [sys.executable, "-B", "-X", "utf8", "-m", "reproduction.train_stage", spec["script"],
               "--config", cfg_path.as_posix(), "--run-dir", run_dir.as_posix(),
               "--device", device]
    if spec["budget"] is not None:
        command += ["--until-tokens", str(spec["budget"])]
    if stage == "poetry_small_batch":
        command += ["--parent-sha256", digest(ROOT / parent) if (ROOT / parent).exists() else "<parent SHA>"]
    elif resume:
        command += ["--resume"]
    return cfg, command, run_dir, parent


def execute(stage, device, resume):
    cfg, command, run_dir, parent = recipe(stage, device, resume)
    target = ROOT / run_dir
    if parent is not None and not (ROOT / parent).is_file():
        raise FileNotFoundError(f"Run {STAGES[stage]['parent']} first: {parent}")
    if parent is not None:
        from poetrygpt.runtime import load_checkpoint
        state = load_checkpoint(ROOT / parent)
        previous = STAGES[stage]["parent"]
        expected = {"pretrain": 630193557, "pretrain_finish": 680193557,
                    "poetry": 1615779, "poetry_small_batch": 3231558, "ending_plan": 400000}[previous]
        if state["tokens_seen"] != expected or state.get("lineage", {}).get("origin") != "random_initialization":
            raise ValueError("Parent has an unfinished budget or a different initialization lineage")
        del state
    cfg_path = ROOT / "reproduction/generated" / (stage + ".json")
    if target.exists() and any(target.iterdir()):
        if not resume:
            raise FileExistsError(f"Existing replay preserved: {run_dir}; use --resume")
        if not cfg_path.is_file() or read(cfg_path) != cfg:
            raise ValueError("Resume recipe differs from the saved recipe")
        if not (target / "last.pt").is_file():
            raise FileNotFoundError("Resume requires last.pt")
    else:
        if resume:
            raise FileNotFoundError("No replay to resume")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="Show the selected training path, without execution")
    run = sub.add_parser("stage", help="Print a stage command; --execute starts it")
    run.add_argument("name", choices=STAGES)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "list":
        print('SELECTED MODEL - training order:')
        for name, spec in STAGES.items():
            print(f"{name:22} {spec['description']}")
        return
    cfg, command, _, _ = recipe(args.name, args.device, args.resume)
    print(json.dumps({"recipe": cfg, "command": command}, ensure_ascii=False, indent=2))
    if args.execute:
        execute(args.name, args.device, args.resume)


if __name__ == "__main__":
    main()

"""Grid of runs (seeds × fractions × variants) for one arm; resumable via MLflow."""

from __future__ import annotations

import argparse
import sys
import traceback

from shared.config import load_config
from shared.runner import run, run_name, tracker_from_cfg


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True)
    p.add_argument("--stage", choices=["dev", "final"], default="dev")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--fractions", type=float, nargs="+", default=None)
    p.add_argument("--variants", nargs="*", default=[None])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.arm, args.set)
    seeds = args.seeds or cfg["seeds"]
    fractions = args.fractions or cfg["fractions"]
    if not cfg["arm"].get("uses_fractions", True):
        fractions = [1.0]  # a0_majority, a1, a5 do not train on fractions
    done = tracker_from_cfg(cfg).completed_run_names(args.arm)

    failed = []
    for variant in args.variants:
        for fraction in fractions:
            for seed in seeds:
                name = run_name(args.arm, args.stage, seed, fraction, variant)
                if name in done:
                    print(f"skip (done): {name}")
                    continue
                print(f"run: {name}", flush=True)
                try:
                    run(args.arm, args.stage, seed, fraction, args.set, variant, args.save_model)
                except Exception:
                    traceback.print_exc()
                    failed.append(name)
    if failed:
        print(f"{len(failed)} run(s) failed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Single run: one arm, one stage, one seed, one fraction (REQUIREMENTS §4.5)."""

from __future__ import annotations

import argparse
import json

from shared.runner import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, help="folder name under arms/, e.g. a0_majority")
    p.add_argument("--stage", choices=["dev", "final"], default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--variant", default=None, help="optional run-name suffix, read by the arm")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    p.add_argument("--save-model", action="store_true", help="save weights (opt-in)")
    args = p.parse_args(argv)
    metrics = run(
        args.arm, args.stage, args.seed, args.fraction, args.set, args.variant, args.save_model
    )
    keep = ("accuracy", "macro_f1")
    headline = {k: round(v, 4) for k, v in metrics.items() if k.split("/")[-1] in keep}
    print(json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run exactly one baseline trial (same trial_idx / seed convention as eval/evaluate.py).

Example — only “trial 1” (second trial), no need to run trial 0 first:
  python eval/run_single_trial.py --trial 1

Default: seed = 42 + trial (same as eval/evaluate.py).

- **trial 0** vs **trial 1** → **different** cereal poses (different `trial_idx` / seed).
- **trial 0** run twice → **same** pose (reproducible). Use **`--stochastic`** only if you want a new pose on every run (not reproducible).
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from src.baseline import run_baseline


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run a single run_baseline trial (matches evaluate.py seed = 42 + trial)."
    )
    p.add_argument(
        "--trial",
        type=int,
        default=0,
        help="0-based trial index (evaluate.py uses trial 0..n-1; seed defaults to 42+trial).",
    )
    p.add_argument("--condition", default="feedback", help="e.g. feedback, feedback_3, baseline")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed (default: 42 + trial). Ignored if --stochastic.",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Pass seed=None to run_baseline: new random cereal pose each run (not reproducible).",
    )
    p.add_argument(
        "--instruction",
        default="pick the cereal",
        help="Language instruction.",
    )
    p.add_argument(
        "--placement-only",
        action="store_true",
        help="Only randomize/apply cereal pose then exit (no OWL-ViT, grasp, or Gemini).",
    )
    p.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Set BASELINE_SKIP_GEMINI=1: run full trial but never call Gemini on failure.",
    )
    args = p.parse_args()
    if args.placement_only:
        os.environ["BASELINE_CEREAL_PLACEMENT_ONLY"] = "1"
    if args.skip_gemini:
        os.environ["BASELINE_SKIP_GEMINI"] = "1"

    if args.stochastic:
        seed = None
    else:
        seed = args.seed if args.seed is not None else 42 + int(args.trial)

    if args.placement_only:
        processor = None
        model = None
        device = torch.device("cpu")
        print("placement-only: skipping OWL-ViT load.")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading OWL-ViT on {device}...")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    print(
        f"--- Single trial: trial_idx={args.trial}, seed={seed!r}, "
        f"stochastic={args.stochastic}, placement_only={args.placement_only}, "
        f"skip_gemini={args.skip_gemini}, condition={args.condition} ---"
    )
    run_baseline(
        args.instruction,
        condition=args.condition,
        trial_idx=args.trial,
        seed=seed,
        processor=processor,
        model=model,
        device=device,
    )


if __name__ == "__main__":
    main()

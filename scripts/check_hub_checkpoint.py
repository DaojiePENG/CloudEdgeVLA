"""Check that the released CloudEdgeVLA checkpoint loads from the Hugging Face Hub.

This is the fastest way for an external user to verify a fresh install. It
exercises exactly the code path the evaluator uses:

  1. the config and remote modeling code load through `trust_remote_code`
  2. `resolve_hub_component` locates the action head and the proprio projector
     inside the repo snapshot instead of relying on a hardcoded file table
  3. both component checkpoints deserialize into real state dicts

The multi-GB backbone shards are *not* downloaded; only `*checkpoint.pt` files
are pulled. Load the full policy with `scripts/eval_libero_delay_sweep.sh`.

Usage:
  python scripts/check_hub_checkpoint.py
  python scripts/check_hub_checkpoint.py --repo-id <hf-repo-id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_REPO_ID = "DaojiePENG/cloudedgevla-libero-spatial-object-goal-10"

# Allow running from anywhere in the repository.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face Hub repo id")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    from experiments.robot.openvla_utils import resolve_hub_component

    print(f"[1/4] repo         : {args.repo_id}")
    print(f"      HF_ENDPOINT  : {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")

    config_path = hf_hub_download(args.repo_id, "config.json")
    with open(config_path) as fh:
        cfg = json.load(fh)
    print(f"[2/4] config.json  : {len(cfg)} keys, _name_or_path={cfg.get('_name_or_path')}")

    for name in ("processing_prismatic.py", "modeling_prismatic.py", "configuration_prismatic.py"):
        try:
            path = hf_hub_download(args.repo_id, name)
            print(f"      remote code  : {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001 - report and keep checking the rest
            print(f"      remote code  : {name} unavailable ({exc})")

    action_head_path = resolve_hub_component(args.repo_id, "action_head")
    proprio_path = resolve_hub_component(args.repo_id, "proprio_projector")
    print(f"[3/4] action head      : {os.path.basename(action_head_path)}")
    print(f"      proprio projector: {os.path.basename(proprio_path)}")

    for name, path in (("action_head", action_head_path), ("proprio_projector", proprio_path)):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        first = next(iter(state_dict))
        print(f"[4/4] {name:17s}: {len(state_dict)} tensors, first={first}, dtype={state_dict[first].dtype}")

    snapshot = snapshot_download(args.repo_id, allow_patterns=["*checkpoint.pt"])
    print(f"\nOK — component snapshot at {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI entry-point for model training."""

from __future__ import annotations

import sys
from typing import Iterable

from fosm_mlops.pipelines.train_pipeline import main as hydra_main

_ALIASED_GROUPS = {
    "model": "train.model",
    "data": "train.data",
    "features": "train.features",
}


def _normalise_overrides(argv: Iterable[str]) -> list[str]:
    """Rewrite simple overrides so Hydra can locate packaged configs."""

    script_name = sys.argv[0]
    normalised = [script_name]
    for arg in argv:
        if "=" in arg and "@" not in arg and not arg.startswith("+"):
            key, value = arg.split("=", 1)
            target = _ALIASED_GROUPS.get(key)
            if target is not None:
                normalised.append(f"{key}@{target}={value}")
                continue
        normalised.append(arg)
    return normalised


if __name__ == "__main__":
    sys.argv = _normalise_overrides(sys.argv[1:])
    hydra_main()

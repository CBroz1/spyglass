"""DeepLabCut Model Zoo catalog lookup and weight fetching.

Two consumers, one source of truth:

- ``Model.load`` / ``Model.load_from_dlc_zoo`` — resolve a string against the
  catalog to decide whether it names a zoo model, then fetch its weights.
- ``maintenance_scripts`` / tests — pre-fetch weights so a test run never
  downloads implicitly.

Lives under ``position/v2/utils/`` rather than the tool-agnostic
``position/utils/`` because the zoo is DLC-3-only: the catalog API
(``dlclibrary.get_available_datasets``) and the snapshot layout both come from
the PyTorch engine, and V1's TensorFlow pipeline has no equivalent.

Two facts shape the API:

**Backbones are per-dataset.** ``superanimal_bird`` offers only ``resnet_50``
and ``superanimal_humanbody`` only ``rtmpose_x``, so a single hardcoded default
is wrong for half the catalog. :func:`default_backbone` reads the catalog.

**Weights are large and cached per environment.** One model is ~646 MB
(152 MB pose + 494 MB detector for topviewmouse/hrnet_w32), and DLC caches them
*inside the installed deeplabcut package* — so the cache is per conda env and
must be re-fetched after an env rebuild. :func:`snapshot_path` defaults to
``download=False`` so callers can check before committing to a download.

Run as a script to inspect or populate the cache::

    python -m spyglass.position.v2.utils.fetch_dlc_zoo            # status
    python -m spyglass.position.v2.utils.fetch_dlc_zoo superanimal_topviewmouse
    python -m spyglass.position.v2.utils.fetch_dlc_zoo --all
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from spyglass.utils.logging import logger


def zoo_catalog() -> Dict[str, List[str]]:
    """Return ``{dataset: [backbone, ...]}`` from dlclibrary.

    Returns
    -------
    dict[str, list[str]]
        Empty when DeepLabCut 3.x is unavailable, so callers can treat "not a
        zoo name" and "no zoo support installed" alike.
    """
    try:
        from dlclibrary import get_available_datasets, get_available_models
    except ImportError:  # pragma: no cover - depends on the env
        logger.debug("dlclibrary unavailable; DLC Model Zoo lookup disabled")
        return {}

    return {
        dataset: list(get_available_models(dataset))
        for dataset in get_available_datasets()
    }


def is_zoo_model(name: str) -> bool:
    """Whether *name* is a DLC Model Zoo dataset.

    Used by ``Model.load`` to decide whether a string that is not an existing
    path should be looked up in the catalog.
    """
    return str(name) in zoo_catalog()


def default_backbone(dataset: str) -> str:
    """First backbone *dataset* offers.

    Raises
    ------
    ValueError
        If *dataset* is not in the catalog.
    """
    catalog = zoo_catalog()
    if dataset not in catalog:
        raise ValueError(
            f"{dataset!r} is not a DLC Model Zoo dataset. "
            f"Available: {sorted(catalog)}"
        )
    return catalog[dataset][0]


def resolve_backbone(dataset: str, model_name: Optional[str] = None) -> str:
    """Validate *model_name* against *dataset*, or pick its default.

    Raises
    ------
    ValueError
        If *dataset* is unknown, or does not offer *model_name*.
    """
    catalog = zoo_catalog()
    if dataset not in catalog:
        raise ValueError(
            f"{dataset!r} is not a DLC Model Zoo dataset. "
            f"Available: {sorted(catalog)}"
        )
    if model_name is None:
        return catalog[dataset][0]
    if model_name not in catalog[dataset]:
        raise ValueError(
            f"{dataset!r} has no backbone {model_name!r}; "
            f"it offers {catalog[dataset]}"
        )
    return model_name


def snapshot_path(
    dataset: str, model_name: Optional[str] = None, download: bool = False
) -> Path:
    """Path to a SuperAnimal snapshot, fetching it only when asked.

    Parameters
    ----------
    dataset : str
        Zoo dataset, e.g. ``'superanimal_topviewmouse'``.
    model_name : str, optional
        Backbone; defaults to whatever the dataset offers first.
    download : bool, optional
        Fetch when absent. Default False -- the path is deterministic, so
        callers can test ``.exists()`` without committing to ~646 MB.

    Returns
    -------
    pathlib.Path
        Inside the installed deeplabcut package, hence per conda environment.
    """
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
        get_super_animal_snapshot_path,
    )

    model_name = resolve_backbone(dataset, model_name)
    return get_super_animal_snapshot_path(
        dataset, model_name, download=download
    )


def is_cached(dataset: str, model_name: Optional[str] = None) -> bool:
    """Whether the snapshot is already on disk for this environment."""
    try:
        return snapshot_path(dataset, model_name, download=False).exists()
    except (ImportError, ValueError):  # pragma: no cover - env dependent
        return False


def fetch(dataset: str, model_name: Optional[str] = None) -> Path:
    """Ensure the snapshot is present, downloading if needed.

    Returns
    -------
    pathlib.Path
        Path to the snapshot.
    """
    model_name = resolve_backbone(dataset, model_name)
    if is_cached(dataset, model_name):
        logger.info(f"Zoo weights already cached: {dataset} / {model_name}")
        return snapshot_path(dataset, model_name, download=False)

    logger.info(
        f"Downloading zoo weights for {dataset} / {model_name}. "
        "This is a few hundred MB and may take several minutes."
    )
    return snapshot_path(dataset, model_name, download=True)


# --------------------------------------------------------------------------
# CLI -- pre-fetch so a test run never downloads implicitly
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    """Inspect or populate the zoo weight cache."""
    parser = argparse.ArgumentParser(
        description="Inspect or populate the DLC Model Zoo weight cache."
    )
    parser.add_argument("dataset", nargs="?", help="zoo dataset to fetch")
    parser.add_argument("--all", action="store_true", help="fetch every model")
    parser.add_argument(
        "--model-name",
        default=None,
        help="backbone; defaults to the first the dataset offers",
    )
    args = parser.parse_args(argv)

    catalog = zoo_catalog()
    if not catalog:
        print(
            "DeepLabCut 3.x not installed -- use the `pv2` env", file=sys.stderr
        )
        return 1

    if args.dataset and args.dataset not in catalog:
        print(
            f"unknown dataset {args.dataset!r}; choose from {sorted(catalog)}",
            file=sys.stderr,
        )
        return 1

    targets = (
        list(catalog) if args.all else ([args.dataset] if args.dataset else [])
    )

    if not targets:  # no args: report cache state, download nothing
        print("cache status:")
        for dataset, backbones in catalog.items():
            backbone = args.model_name or backbones[0]
            if args.model_name and args.model_name not in backbones:
                print(f"  [n/a    ] {dataset:26} (offers {backbones})")
                continue
            mark = "cached " if is_cached(dataset, backbone) else "MISSING"
            name = snapshot_path(dataset, backbone).name
            print(f"  [{mark}] {dataset:26} {backbone:16} {name}")
        print("\nPass a dataset name or --all to fetch.")
        return 0

    for dataset in targets:
        try:
            backbone = resolve_backbone(dataset, args.model_name)
        except ValueError as err:
            print(err, file=sys.stderr)
            return 1
        state = "cached" if is_cached(dataset, backbone) else "fetch "
        print(f"[{state}]  {dataset} / {backbone}")
        fetch(dataset, backbone)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

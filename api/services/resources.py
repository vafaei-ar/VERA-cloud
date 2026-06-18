"""
Local resource layer (A.8) — geographic + need lookup.

Returns curated local resources (transportation, meals, rehab, devices, support)
keyed by region (county) and need. This is an OPT-IN, clearly separate feature —
it is not part of the core check-in flow, so it does not clutter the assessment.

Framing: resources are presented as INFORMATION ONLY — what exists and how to look
into it — never as an endorsement or a promise of insurance coverage (per Phil).

Data lives in config/resources.yaml so the care team can maintain it without code
changes. No Azure / network dependencies — safe to import and unit test.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)

NEED_CATEGORIES = ("transportation", "meals", "rehab", "devices", "support")

INFO_ONLY_DISCLAIMER = (
    "This is general information about what may exist in your area, not a "
    "recommendation or endorsement. It does not promise that any service is "
    "available to you or covered by your insurance. Please check with each "
    "service and your care team."
)


def _resources_path() -> Path:
    default = Path(__file__).resolve().parent.parent.parent / "config" / "resources.yaml"
    return Path(os.environ.get("RESOURCES_PATH", str(default)))


def _load_data() -> Dict[str, Any]:
    path = _resources_path()
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to load resources from {path}: {e}")
        return {}


def _normalize_region(region: Optional[str]) -> str:
    return (region or "").strip().lower()


def list_regions() -> List[Dict[str, str]]:
    """Return available regions with their display names."""
    data = _load_data()
    out = []
    for key, block in (data.get("regions") or {}).items():
        out.append({"region": key, "display_name": block.get("display_name", key)})
    return out


def lookup_resources(region: Optional[str], need: Optional[str] = None) -> Dict[str, Any]:
    """Look up local resources by region (county) and optional need category.

    Falls back to the `default` block when the region is unknown. If `need` is
    given but invalid, returns an empty resource list with a note. Always includes
    the information-only disclaimer.
    """
    data = _load_data()
    region_key = _normalize_region(region)
    regions = data.get("regions") or {}
    block = regions.get(region_key)
    used_fallback = False
    if block is None:
        block = data.get("default") or {}
        used_fallback = True

    result: Dict[str, Any] = {
        "region": region_key or None,
        "display_name": block.get("display_name", "General"),
        "used_fallback": used_fallback,
        "disclaimer": INFO_ONLY_DISCLAIMER,
        "resources": {},
    }

    if need:
        n = need.strip().lower()
        if n not in NEED_CATEGORIES:
            result["note"] = f"Unknown need '{need}'. Valid needs: {', '.join(NEED_CATEGORIES)}."
            return result
        result["resources"][n] = block.get(n, [])
    else:
        for n in NEED_CATEGORIES:
            if block.get(n):
                result["resources"][n] = block.get(n)
    return result

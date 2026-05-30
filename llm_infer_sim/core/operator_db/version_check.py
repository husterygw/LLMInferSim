"""Operator-DB framework-version precheck.

The measured operator DB is captured under one vLLM version
(`<root>/<hw>/vllm-<version>/`). Its kernel latencies are only valid for that
version: `framework_version` is part of the OperatorSignature, so a runtime on
a different vLLM version silently misses every record (observed: running 0.19.0
against a 0.19.1 DB → 0 hits, even for ops with no canonical口径 bug).

This module makes that mismatch loud instead of silent:
- default (fail-closed): raise `OperatorDBVersionMismatch`.
- `ignore_mismatch=True` (debug / `--ignore-framework-version`): warn + continue.

CI / bench should never default to ignoring — a version skew means the DB hits
you think you're measuring are actually roofline fallbacks.
"""
from __future__ import annotations

import warnings


class OperatorDBVersionMismatch(RuntimeError):
    """Runtime vLLM version != operator-DB framework_version (fail-closed)."""


def current_runtime_vllm_version() -> str | None:
    """Runtime vLLM version, or None if vLLM is not importable.

    Imported lazily so this core module keeps no hard dependency on the vLLM
    adapter layer.
    """
    try:
        import vllm  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return getattr(vllm, "__version__", None)


def normalize_version(v: str | None) -> str | None:
    """Strip build/local metadata so '0.19.1+cu128' == '0.19.1'."""
    if v is None:
        return None
    return str(v).strip().lstrip("vV").split("+", 1)[0]


def verify_db_framework_version(
    db_version: str,
    runtime_version: str | None = None,
    *,
    ignore_mismatch: bool = False,
    source: str = "",
) -> str | None:
    """Check the loaded DB's framework_version against the runtime vLLM version.

    Returns the resolved runtime version (may be None if undeterminable).

    - match            -> return runtime_version.
    - runtime unknown  -> warn (can't verify) and return None.
    - mismatch         -> raise OperatorDBVersionMismatch, unless
                          ``ignore_mismatch`` (then warn and continue).
    """
    if runtime_version is None:
        runtime_version = current_runtime_vllm_version()

    where = f" ({source})" if source else ""

    if runtime_version is None:
        warnings.warn(
            f"operator DB version check skipped{where}: runtime vLLM version "
            f"could not be determined (DB framework_version={db_version!r}).",
            stacklevel=2,
        )
        return None

    if normalize_version(db_version) == normalize_version(runtime_version):
        return runtime_version

    msg = (
        f"operator DB framework_version={db_version!r} does not match runtime "
        f"vLLM version={runtime_version!r}{where}. The DB's measured kernel "
        f"latencies are version-specific; signatures embed framework_version, "
        f"so every lookup will silently miss and fall back to roofline. "
        f"Re-collect on the runtime version or run on the DB's version."
    )
    if ignore_mismatch:
        warnings.warn(msg + " [ignored via ignore_framework_version]", stacklevel=2)
        return runtime_version
    raise OperatorDBVersionMismatch(msg)

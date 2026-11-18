"""VESSL fixtures and skip rule (DESIGN.md §17.6.1, §17.6.2, §17.6.3).

Every test under ``tests/vessl/`` needs ``ILORA_VESSL=1`` and CUDA. Otherwise the
collection hook below marks it skipped with the reason ``VESSL_REQUIRED``
before any fixture runs; it also adds the ``vessl`` marker. Fixtures:
``llm_path``, ``llama_bundle``, ``llama_model`` (session), ``run_dir_env(name)``
(``ILORA_RUN_<NAME>`` run folders named in the RUNBOOK, §18.6.6) and
``measurements`` (values recorded without judgement, written at the end of the
session to ``runs/rebuild/vessl_tests_<UTC>/measurements.json`` with M0
``create_run_dir``) and ``lightning`` (PyTorch Lightning of the env for the
M10 14.8 #18 lineage check; skips with ``SKIP_NO_PL`` when it cannot be imported).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

VESSL_DIR = Path(__file__).resolve().parent

# Documented skip reasons of the VESSL tests (§17.6.2, §17.6.3, §17.8 #2).
VESSL_REQUIRED = "VESSL_REQUIRED"
SKIP_NO_RUN = "run not available (set ILORA_RUN_<NAME> to the run folder, DESIGN 18.6.6)"
SKIP_NO_LLM = "LLM_PATH not available (set LLM_PATH to the Llama-2-7b-hf folder)"
SKIP_NO_PL = ("pytorch_lightning not importable (U requirements pin 1.8.6; DESIGN 14.8 #18 records the reason in "
              "measurements.json)")
DOCUMENTED_SKIP_REASONS = (VESSL_REQUIRED, SKIP_NO_RUN, SKIP_NO_LLM, SKIP_NO_PL)


def vessl_available() -> bool:
    """``ILORA_VESSL == "1"`` and CUDA available."""
    if os.environ.get("ILORA_VESSL") != "1":
        return False
    import torch

    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    ready: Optional[bool] = None
    for item in items:
        path = Path(str(item.path)).resolve()
        if VESSL_DIR not in path.parents:
            continue
        item.add_marker(pytest.mark.vessl)
        if ready is None:
            ready = vessl_available()
        if not ready:
            item.add_marker(pytest.mark.skip(reason=VESSL_REQUIRED))


@pytest.fixture(scope="session")
def llm_path() -> Path:
    """``LLM_PATH`` (Llama-2-7b-hf, fp16)."""
    value = os.environ.get("LLM_PATH")
    if not value or not Path(value).is_dir():
        pytest.skip(SKIP_NO_LLM)
    return Path(value)


@pytest.fixture(scope="session")
def llama_bundle(llm_path: Path, paper_cfg: Any) -> Any:
    """M5 ``load_tokenizer(LLM_PATH, paper_cfg.batching.tokenizer_sha256)``."""
    from ilora.batching import load_tokenizer

    return load_tokenizer(llm_path, paper_cfg.batching.tokenizer_sha256)


@pytest.fixture(scope="session")
def llama_model(llama_bundle: Any, full_data: Any, paper_cfg: Any, llm_path: Path) -> Any:
    """M9 ``build_model(paper_cfg, llama_bundle, catalog, llm_path=LLM_PATH)`` (cuda:0, fp16)."""
    from ilora.model import build_model

    return build_model(paper_cfg, llama_bundle, full_data.catalog, llm_path=llm_path)


@pytest.fixture
def run_dir_env() -> Callable[[str], Path]:
    """Factory ``run_dir_env(name) -> Path`` of ``ILORA_RUN_<NAME>`` (e.g. ``E2``, ``E2_EXPORT``, ``CLUSTER``)."""
    def get(name: str) -> Path:
        value = os.environ.get(f"ILORA_RUN_{name.upper()}")
        if not value or not Path(value).is_dir():
            pytest.skip(SKIP_NO_RUN)
        return Path(value)

    return get


class Measurements(dict):
    """Measured values of the VESSL tests: recorded, not judged (§17.1, §17.6.2, TBD values)."""

    def write(self, root: Optional[Path] = None) -> Path:
        """Write ``measurements.json`` into a new ``<root>/vessl_tests_<UTC>/`` (M0 ``create_run_dir``)."""
        from ilora import run

        run_dir = run.create_run_dir("vessl_tests", root=Path(root) if root is not None else run.RUNS_ROOT)
        run.write_json_atomic(run_dir / "measurements.json",
                              {"written_utc": run.utc_now().isoformat(), "measurements": dict(self)})
        return run_dir


@pytest.fixture(scope="session")
def measurements() -> Any:
    values = Measurements()
    yield values
    if values:
        values.write()


@pytest.fixture(scope="session")
def lightning(measurements: Any) -> Any:
    """``pytorch_lightning`` of the VESSL env (M10 14.8 #18, lineage check only); when it cannot be imported the
    reason goes into ``measurements`` and the test skips with ``SKIP_NO_PL``."""
    try:
        import pytorch_lightning
    except Exception as exc:  # noqa: BLE001 - recorded, then skipped with the documented reason
        measurements["pl_native_amp_scaler"] = {"status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}"}
        pytest.skip(SKIP_NO_PL)
    return pytorch_lightning

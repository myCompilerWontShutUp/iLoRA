"""VESSL 환경 시험 (DESIGN.md §17.6.9, §18.8 #9, §17.8 #7). VESSL_REQUIRED: skipped locally."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.vessl


@pytest.mark.criterion("M14-9")
def test_check_env_passes(paper_cfg, llm_path, tmp_path, measurements):
    """R4 check-env passes in the R1 env (transformers file integrity included); pip freeze is kept."""
    from ilora import commands, run

    run_dir = commands.run_command("check-env", paper_cfg, [], profile="paper_spec", root=tmp_path,
                                   llm_path=llm_path, argv=["test_check_env_passes"])
    meta = run.read_meta(run_dir)
    report = json.loads((run_dir / "check_env.json").read_text(encoding="utf-8"))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                            check=True).stdout
    measurements["check_env"] = {"status": meta["status"], "environment": meta["environment"],
                                 "assets": meta["assets"], "report": report, "pip_freeze": freeze.splitlines()}
    assert meta["status"] == "completed"
    assert report["passed"] is True
    assert run.transformers_integrity_problems() == []


@pytest.mark.criterion("M13-7")
def test_measurements_writer(measurements, tmp_path):
    """The measurements writer puts JSON into a new folder made by M0 create_run_dir."""
    from ilora import run

    measurements["writer_probe"] = {"python": sys.version.split()[0]}
    first = measurements.write(root=tmp_path)
    assert first.parent == tmp_path and first.name.startswith("vessl_tests_")
    data = json.loads((first / "measurements.json").read_text(encoding="utf-8"))
    assert data["measurements"]["writer_probe"] == measurements["writer_probe"]
    with pytest.raises(FileExistsError):
        run.create_run_dir("vessl_tests", stamp=first.name[len("vessl_tests_"):], root=tmp_path)


@pytest.mark.criterion("M1-12")
def test_data_on_vessl(repo_root, paper_cfg, measurements):
    """VESSL work tree (§5.8 #12): the four data files hash to run.EXPECTED_INPUTS, pandas 1.5.3 and
    numpy 1.24.2, and M1 gives the same content fingerprints and sample counts as locally."""
    import numpy
    import pandas
    from ilora import data, run

    hashes = {}
    for name in (*data.SPLIT_FILES.values(), data.ITEM_FILE):
        rel = f"{data.DATA_DIR}/{name}"
        hashes[rel] = run.content_sha256(repo_root / rel, run.EXPECTED_INPUTS[rel][1])
    fingerprints = {}
    for split, name in data.SPLIT_FILES.items():
        rel = f"{data.DATA_DIR}/{name}"
        if hashes[rel] == run.EXPECTED_INPUTS[rel][0]:                  # unpickle verified files only
            frame = data.project_frame(pandas.read_pickle(repo_root / rel), split)
            fingerprints[split] = data.frame_fingerprint(frame)
    record = {"numpy": numpy.__version__, "pandas": pandas.__version__, "sha256": hashes,
              "fingerprints": fingerprints}
    measurements["m1_data"] = record
    assert numpy.__version__ == "1.24.2" and pandas.__version__ == "1.5.3"
    assert all(hashes[rel] == run.EXPECTED_INPUTS[rel][0] for rel in hashes), hashes
    assert fingerprints == data.EXPECTED_FINGERPRINTS
    full = data.load_dataset(paper_cfg, repo_root)
    record["samples"] = {split: len(full.samples(split)) for split in data.SPLITS}
    assert record["samples"] == data.EXPECTED_SAMPLES
    assert len({s.user_id for s in full.train}) == data.TRAIN_USERS


@pytest.mark.criterion("M2-9")
def test_numpy_and_golden(measurements):
    """VESSL numpy is 1.24.2 and candidates.self_check() passes (§6.8 #9)."""
    import numpy
    from ilora import candidates

    measurements["m2_numpy"] = numpy.__version__
    assert numpy.__version__ == "1.24.2"
    candidates.self_check()


@pytest.mark.criterion("M5-11")
def test_llm_tokenizer_sha256(llm_path, paper_cfg, measurements):
    """``$LLM_PATH/tokenizer.model`` has the sha256 of ``batching.tokenizer_sha256`` and passes M5
    ``load_tokenizer`` (ids, fingerprints) (§9.8 #11). The #1-#10 tests are re-run by the VESSL CPU bundle
    with ``ILORA_TEST_TOKENIZER=$LLM_PATH`` (§17.6.1)."""
    from ilora import batching, run

    got = run.content_sha256(llm_path / "tokenizer.model")
    measurements["m5_llm_tokenizer"] = {"tokenizer_model_sha256": got,
                                        "expected": paper_cfg.batching.tokenizer_sha256}
    assert got == paper_cfg.batching.tokenizer_sha256
    bundle = batching.load_tokenizer(llm_path, paper_cfg.batching.tokenizer_sha256)
    assert bundle.ids.vocab_size == len(bundle.tokenizer) == 32005
    assert bundle.ids.new_token_ids == (32000, 32001, 32002, 32003, 32004)
    for text, expected in batching.FINGERPRINTS:
        assert tuple(batching.encode_text(bundle, text)) == expected


@pytest.mark.criterion("M5-12")
def test_workers_8_linux(llama_bundle, full_data, templates, profile_cfg, measurements):
    """paper_spec epoch 0: the first 50 TrainBatch with 8 workers (Linux, fork start) equal those with 0
    workers, every tensor and list (§9.8 #12)."""
    import itertools
    import random
    import time

    import numpy
    import torch

    from ilora import batching, prompts

    n_batches = 50
    results, seconds = {}, {}
    for workers in (0, 8):
        cfg = profile_cfg("paper_spec", f"batching.num_workers={workers}")
        plan = prompts.build_batch_plan(full_data.train, 0, cfg)
        saved = (random.getstate(), numpy.random.get_state(), torch.get_rng_state())
        start = time.perf_counter()
        try:
            results[workers] = list(itertools.islice(
                batching.make_train_loader(plan, full_data, cfg, 0, bundle=llama_bundle, templates=templates),
                n_batches))
        finally:
            random.setstate(saved[0])
            numpy.random.set_state(saved[1])
            torch.set_rng_state(saved[2])
        seconds[workers] = round(time.perf_counter() - start, 3)
    measurements["m5_workers_8_linux"] = {"batches": n_batches, "seconds": seconds}
    assert len(results[0]) == len(results[8]) == n_batches
    for i, (a, b) in enumerate(zip(results[0], results[8])):
        for name in a.__dataclass_fields__:
            x, y = getattr(a, name), getattr(b, name)
            if isinstance(x, torch.Tensor):
                assert x.dtype == y.dtype and torch.equal(x.cpu(), y.cpu()), (i, name)
            else:
                assert x == y, (i, name)
    assert [b.tau for b in results[8]] == list(range(1, n_batches + 1))


@pytest.mark.criterion("M11-21")
def test_decode_rules_llm_tokenizer(llama_bundle, paper_cfg, measurements):
    """M11 #3 and #4 with the ``$LLM_PATH`` tokenizer (§15.8 #21): BOS placeholder and [PAD] after EOS change no
    decoded text, the 8 special ids are dropped, only id 13 decodes to a newline, the newline counterexample gives
    "Star Wars" against "" and the paper_spec generation config stops at EOS 2 (one id) or the maximum length."""
    from ilora import batching, evaluate
    from ilora.model import generation_config_for

    special = (0, 1, 2, 32000, 32001, 32002, 32003, 32004)
    assert sorted(llama_bundle.tokenizer.all_special_ids) == list(special)
    for text in ("Star Wars\n", "\nStar Wars\nHeat\n", "Good Will Hunting\nThe Rock", " \n Toy Story\n"):
        a = batching.encode_text(llama_bundle, text)
        padded = a + [2] + [32000] * 5
        assert evaluate.decode_generated(llama_bundle, [a, [1] + a, padded]) == [text] * 3, text
        assert evaluate.trim_generated(padded, eos_id=2) == tuple(a + [2])
    a = batching.encode_text(llama_bundle, "Star Wars\n")
    mixed = [1, 32001] + a[:1] + [32002, 0, 32003] + a[1:] + [32000, 32004, 2]
    assert evaluate.decode_generated(llama_bundle, [mixed]) == ["Star Wars\n"]
    texts = evaluate.decode_generated(llama_bundle, [[i] for i in range(32000)])
    newline_ids = [i for i, t in enumerate(texts) if "\n" in t]
    measurements["m11_llm_tokenizer_newline_ids"] = newline_ids
    assert newline_ids == [13]
    gen = [29871, 13, 16213, 9149, 13, 3868, 271, 13]
    assert batching.encode_text(llama_bundle, "\nStar Wars\nHeat\n") == gen
    assert evaluate.first_line(evaluate.decode_generated(llama_bundle, [gen])[0]) == "Star Wars"
    assert evaluate.first_line(evaluate.decode_generated(llama_bundle, [gen[:2]])[0]) == ""
    gc = generation_config_for(paper_cfg.model, llama_bundle.ids)
    assert gc.eos_token_id == 2 and isinstance(gc.eos_token_id, int) and gc.max_time is None
    assert (gc.do_sample, gc.num_beams, gc.max_new_tokens, gc.pad_token_id) == (False, 1, 64, 32000)


@pytest.mark.criterion("M10-18")
def test_pl_native_amp_scaler_defaults(lightning, measurements):
    """Lineage of the GradScaler values (§14.4, 14.8 #18): PyTorch Lightning 1.8.6 (U requirements) builds the native
    AMP scaler of ``--precision 16`` without arguments, so its (init_scale, growth_factor, backoff_factor,
    growth_interval) are the torch 2.0.0 defaults (65536, 2.0, 0.5, 2000) that ``train.scaler_*`` copies. The values
    and the installed version are recorded; they do not affect any result."""
    from pytorch_lightning.plugins.precision import NativeMixedPrecisionPlugin

    scaler = NativeMixedPrecisionPlugin(16, "cuda").scaler
    values = (scaler.get_scale(), scaler.get_growth_factor(), scaler.get_backoff_factor(),
              scaler.get_growth_interval())
    measurements["pl_native_amp_scaler"] = {"status": "measured", "pytorch_lightning": lightning.__version__,
                                            "enabled": scaler.is_enabled(), "values": list(values)}
    assert scaler.is_enabled()
    assert values == (65536.0, 2.0, 0.5, 2000)

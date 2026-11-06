"""VESSL SASRec 시험 (DESIGN.md §7.8 #12, §17.6.9 M3-12). VESSL_REQUIRED: skipped locally.

The converted asset (``runs/rebuild/assets/sasrec`` made by ``tools/convert_sasrec.py`` on the
VESSL machine, RUNBOOK) holds CPU reference outputs of the original code. On ``cuda:0`` the
encoder must stay within 1e-4 of them with ``strict_fp32`` (TF32 off), repeat bit for bit, leave
both TF32 flags as the caller had them, and look items up bit for bit. The difference with TF32
switched on is measured once and only recorded (no requirement, TBD, §17.6.10).
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import pytest
import torch

pytestmark = pytest.mark.vessl


@pytest.mark.criterion("M3-12")
def test_cuda_matches_cpu_reference(sasrec_state_path, sasrec_reference_path, paper_cfg, monkeypatch, measurements):
    from ilora import sasrec
    from ilora.data import HISTORY_LEN, NUM_ITEMS, PAD_ID, ItemCatalog

    device = torch.device("cuda:0")
    catalog = ItemCatalog(titles=tuple(f"item {i}" for i in range(NUM_ITEMS)), title_mode="code",
                          num_items=NUM_ITEMS, pad_id=PAD_ID)
    encoder = sasrec.SASRecEncoder.from_config(paper_cfg, catalog, HISTORY_LEN).to(device)
    reference = torch.load(sasrec_reference_path, map_location="cpu", weights_only=True)
    record: dict = {"tolerance": sasrec.CUDA_REFERENCE_ATOL, "device_name": torch.cuda.get_device_name(device)}
    measurements["m3_cuda_reference"] = record
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    record["tf32_flags_at_start"] = list(saved)
    try:
        # Both flags on before every call: a call that left them off (or did not switch them off inside)
        # would show in the values after the call or in the TF32 measurement below.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            report = sasrec.verify_against_reference(encoder, sasrec_reference_path, device)
        except sasrec.SASRecError as exc:
            record["verify"] = getattr(exc, "report", str(exc))
            raise
        record["verify"] = report
        flags_after = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        record["tf32_flags_after_call"] = list(flags_after)
        assert flags_after == (True, True)
        assert report["passed"] and report["max_abs_diff"] <= sasrec.CUDA_REFERENCE_ATOL
        assert report["item_emb"]["bitwise"]                       # CUDA embed_items = CPU, bit for bit

        hist = reference["hist_ids"][:4096].to(device)
        lens = reference["len_seq"][:4096].to(device)
        first, second = encoder(hist, lens), encoder(hist, lens)
        record["repeat_bitwise"] = bool(torch.equal(first, second))
        assert record["repeat_bitwise"]

        @contextlib.contextmanager
        def autocast_off_tf32_on(dev: torch.device) -> Iterator[None]:
            # strict_fp32 without its TF32 switch: autocast off, TF32 flags as set above (True)
            with torch.autocast(device_type=torch.device(dev).type, enabled=False):
                yield

        monkeypatch.setattr(sasrec, "strict_fp32", autocast_off_tf32_on)
        all_rows, all_lens = reference["hist_ids"].tolist(), reference["len_seq"].tolist()
        worst = 0.0
        for split in ("train", "val", "test"):                   # one split per call, as the reference
            rows = [i for i, sid in enumerate(reference["sample_ids"]) if sid.startswith(split + ":")]
            histories = [all_rows[i][:all_lens[i]] for i in rows]
            z = encoder.encode_all(histories, chunk=reference["chunk"], num_threads=reference["num_threads"])
            worst = max(worst, float((z - reference["z"][rows]).abs().max()))
        record["tf32_on_max_abs_diff"] = worst                     # recorded only (TBD)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved

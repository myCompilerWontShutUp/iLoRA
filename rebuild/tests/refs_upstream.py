"""공개 코드 동작 사본 (DESIGN.md §17.6.4, §17.4). 대조 전용, 실행 경로에서 쓰지 않음.

Each function copies, line by line, what the released iLoRA code ``U``
(AkaliKong/iLoRA ``28e47371192121b6ced44401d9e149fedfae13de``) or our fork
``F`` (myCompilerWontShutUp/iLoRA ``exp/gate-cluster``, baseline ``6c7ebac``)
does, defects included, so that tests can compare the re-implementation with
the released behaviour. Every function names its source lines. Nothing under
``rebuild/ilora`` imports this file and this file imports no repository code
(DESIGN.md §1.3); it is not a test module.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# First line and HR@1 judgement (M11 #1, M2 #6)
# ----------------------------------------------------------------------------
def first_line_and_hr1(generate: str, cand_titles: Sequence[str], target_title: str) -> tuple[str, bool, bool]:
    """첫 줄과 판정 한 표본. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U model/model_interface.py:89-90`` (decode then ``strip``), ``:165``
    (``generate.strip().split("\\n")[0]`` in ``validation_step``; ``test_step`` :200
    is the same) and ``:468-492`` (``calculate_hr1`` per-sample body).

    ``generate`` is one element of ``batch_decode(generate_ids, skip_special_tokens=True,
    clean_up_tokenization_spaces=False)`` (:89). Returns ``(first_line, valid, correct)``:
    ``valid`` = exactly one lower-cased candidate title is a substring of the lower-cased
    first line (:479-483), ``correct`` = valid and that candidate equals the target (:485).
    """
    output = generate.strip()                                          # :90  outputs=[text.strip() ...]
    first_line = output.strip().split("\n")[0]                         # :165
    gen = first_line.strip().lower().strip()                           # :476
    real = target_title.strip().lower().strip()                        # :477
    cans = [item.strip().lower().strip() for item in cand_titles]      # :478
    gen_cans_list = []                                                 # :479
    for cans_item in cans:                                             # :480
        if cans_item in gen:                                           # :481
            gen_cans_list.append(cans_item)                            # :482
    valid = len(gen_cans_list) == 1                                    # :483
    correct = valid and real == gen_cans_list[0]                       # :485
    return first_line, valid, correct


def code_calculate_hr1(eval_content: Mapping[str, Sequence[Any]]) -> tuple[float, float]:
    """``calculate_hr1`` 전체 (ValidRatio, HR@1 = 정답/유효). 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U model/model_interface.py:468-492``. ``eval_content`` has the lists
    ``generate`` (first lines, :165), ``real`` (target titles) and ``cans`` (candidate
    title lists) as in ``on_validation_batch_end`` (:169-173). The upstream metric is
    ``hr1 * valid_ratio`` (:180-181).
    """
    correct_num = 0                                                    # :469
    valid_num = 0                                                      # :470
    total_num = 0                                                      # :471
    for i, generate in enumerate(eval_content["generate"]):            # :472
        real = eval_content["real"][i]                                 # :473
        cans = eval_content["cans"][i]                                 # :474
        total_num += 1                                                 # :475
        generate = generate.strip().lower().strip()                    # :476
        real = real.strip().lower().strip()                            # :477
        cans = [item.strip().lower().strip() for item in cans]         # :478
        gen_cans_list = []                                             # :479
        for cans_item in cans:                                         # :480
            if cans_item in generate:                                  # :481
                gen_cans_list.append(cans_item)                        # :482
        if len(gen_cans_list) == 1:                                    # :483
            valid_num += 1                                             # :484
            if real == gen_cans_list[0]:                               # :485
                correct_num += 1                                       # :486
    valid_ratio = valid_num / total_num                                # :487
    if valid_num > 0:                                                  # :488
        hr1 = correct_num / valid_num                                  # :489
    else:
        hr1 = 0                                                        # :491
    return valid_ratio, hr1                                            # :492


# ----------------------------------------------------------------------------
# Prompt substitution chain (M4 #3)
# ----------------------------------------------------------------------------
def code_prompt(template: str, history_titles: Sequence[str], cand_titles: Sequence[str], hybrid: bool) -> str:
    """prompt 치환 연쇄. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U data/data_interface.py:43-48`` (hybrid: ``[HistoryEmb]``/``[CansEmb]``)
    and ``:56-63`` (text-only: ``[PH]``). ``template`` is one line of ``prompt/movie.txt``
    after ``splitlines()`` and ``strip()`` (``U data/data_interface.py:182-186``).
    The ``[TargetHere]`` branch (:49-51) is outside the cited lines and not copied: the
    MovieLens templates have no ``[TargetHere]`` (``U prompt/movie.txt:1-3``).
    """
    input_text = template
    if hybrid:
        if '[HistoryHere]' in input_text:                                                         # :43
            insert_prompt = ", ".join([seq_title + ' [HistoryEmb]' for seq_title in history_titles])  # :44
            input_text = input_text.replace('[HistoryHere]', insert_prompt)                       # :45
        if '[CansHere]' in input_text:                                                            # :46
            insert_prompt = ", ".join([can_title + ' [CansEmb]' for can_title in cand_titles])     # :47
            input_text = input_text.replace('[CansHere]', insert_prompt)                          # :48
    else:
        if '[HistoryHere]' in input_text:                                                         # :58
            insert_prompt = ", ".join([seq_title + ' [PH]' for seq_title in history_titles])      # :59
            input_text = input_text.replace('[HistoryHere]', insert_prompt)                       # :60
        if '[CansHere]' in input_text:                                                            # :61
            insert_prompt = ", ".join([can_title + ' [PH]' for can_title in cand_titles])         # :62
            input_text = input_text.replace('[CansHere]', insert_prompt)                          # :63
    return input_text


# ----------------------------------------------------------------------------
# Pair encoding and label mask (M5 #2)
# ----------------------------------------------------------------------------
def code_pair_input_and_labels(tok: Any, prompts: Sequence[str], answers: Sequence[str], *,
                               terminator: str = "\n") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """학습 쌍 인코딩과 label mask. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U data/data_interface.py:69-82`` (answer = target title + terminator,
    ``tokenizer([[prompt, answer], ...])`` with ``padding="longest"``,
    ``return_token_type_ids=True``), ``U model/model_interface.py:316`` (right padding)
    and ``U model/model_interface.py:38-42`` (labels: pad -> -100, then
    ``(token_type_ids == 0)[:, 1:]`` -> -100). With transformers 4.28.0 the slow
    tokenizer's ``token_type_ids`` are one longer than ``input_ids`` and the kept labels
    are the second ``[BOS]`` plus the answer (DESIGN.md §9.5, D18; shape error from 4.29.0).

    ``answers`` are the target titles WITHOUT the terminator (``sample['correct_answer']``);
    ``tok`` is the LlamaTokenizer with the special tokens added (``[PAD]`` = pad).
    Returns ``(input_ids, attention_mask, labels)``, each ``[B, L]``.
    """
    targets_text = [target_text + terminator for target_text in answers]                 # :72
    inputs_pair = [[p, t] for p, t in zip(prompts, targets_text)]                         # :73
    saved_side = tok.padding_side
    tok.padding_side = "right"                                                            # model_interface :316
    try:
        batch_tokens = tok(                                                               # :75-82
            inputs_pair,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True)
    finally:
        tok.padding_side = saved_side
    targets = batch_tokens.input_ids.masked_fill(                                         # model_interface :38-40
        batch_tokens.input_ids == tok.pad_token_id, -100
    )
    targets = targets.masked_fill((batch_tokens.token_type_ids == 0)[:, 1:], -100)        # model_interface :42
    return batch_tokens.input_ids, batch_tokens.attention_mask, targets


# ----------------------------------------------------------------------------
# Behaviour-token insertion loop (M9 #5)
# ----------------------------------------------------------------------------
def wrap_emb_loop(input_embeds: torch.Tensor, input_ids: torch.Tensor, his_item_embeds: torch.Tensor,
                  cans_item_embeds: torch.Tensor, len_seq: torch.Tensor, len_cans: torch.Tensor, *,
                  his_token_id: int, cans_token_id: int) -> torch.Tensor:
    """행동 토큰 삽입 반복. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U model/model_interface.py:453-461`` (``wrap_emb``). Writes the projected
    item embeddings into ``input_embeds`` IN PLACE at the ``[HistoryEmb]`` / ``[CansEmb]``
    positions, history in order up to ``len_seq[i]`` and candidates up to ``len_cans[i]``.
    Defect kept: ``zip`` stops at the shorter side, so a token/embedding count mismatch
    is silently truncated (DESIGN.md §9.5). The ``[ItemEmb]`` path (:462-464) is not copied.
    Returns ``input_embeds`` (the same tensor).
    """
    for i in range(len(len_seq)):                                                              # :453
        if (input_ids[i] == his_token_id).nonzero().shape[0] > 0:                              # :454
            idx_tensor = (input_ids[i] == his_token_id).nonzero().view(-1)                     # :455
            for idx, item_emb in zip(idx_tensor, his_item_embeds[i, :len_seq[i].item()]):      # :456
                input_embeds[i, idx] = item_emb                                                # :457
        if (input_ids[i] == cans_token_id).nonzero().shape[0] > 0:                             # :458
            idx_tensor = (input_ids[i] == cans_token_id).nonzero().view(-1)                    # :459
            for idx, item_emb in zip(idx_tensor, cans_item_embeds[i, :len_cans[i].item()]):    # :460
                input_embeds[i, idx] = item_emb                                                # :461
    return input_embeds


# ----------------------------------------------------------------------------
# Learning-rate schedule (M10 #2)
# ----------------------------------------------------------------------------
def upstream_lr(cur_step: int, cur_epoch: int, max_step: int, *, init_lr: float, min_lr: float,
                warmup_start_lr: float, warmup_steps: int) -> float:
    """한 parameter 그룹의 lr. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U optims.py:20-36`` (``LinearWarmupCosineLRScheduler.step``,
    ``cosine_lr_schedule``, ``warmup_lr_schedule``). Upstream calls it with
    ``cur_step = trainer.global_step``, ``cur_epoch = current_epoch``,
    ``max_step = trainer.max_steps`` (``U model/model_interface.py:136``) and
    ``warmup_steps = max_step // 20`` (``:248``). Warmup happens only in epoch 0 (:22).
    """
    if cur_epoch == 0 and cur_step < warmup_steps:                                        # :22
        # warmup_lr_schedule(cur_step, warmup_start_lr, init_lr)                           # :23, :33-36
        return min(init_lr, warmup_start_lr + (init_lr - warmup_start_lr) * cur_step / max(warmup_steps, 1))
    # cosine_lr_schedule(cur_step - warmup_steps, max_step - warmup_steps, init_lr, min_lr)  # :25, :28-31
    step = cur_step - warmup_steps
    total = max_step - warmup_steps
    return (init_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / total)) + min_lr


def upstream_group_lr_lists(lr: float, lr_decay_min_lr: float,
                            lr_warmup_start_lr: float) -> dict[str, list[float]]:
    """세 그룹 (projector, router, LLM)의 init/min/warmup-start 목록. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U model/model_interface.py:253-267``: the router group gets x0.3 of every
    value, the other two x1 (``variant_released_gate_lr``, D19).
    """
    return {
        "init_lr_list": [lr, lr * 0.3, lr * 1],                                          # :253-257
        "min_lr_list": [lr_decay_min_lr, lr_decay_min_lr * 0.3, lr_decay_min_lr * 1],    # :258-262
        "warmup_start_lr_list": [lr_warmup_start_lr, lr_warmup_start_lr * 0.3, lr_warmup_start_lr * 1],  # :263-267
    }


# ----------------------------------------------------------------------------
# MoE-LoRA expert einsum and gate sum (M8 #2)
# ----------------------------------------------------------------------------
def upstream_scaling(lora_alpha: float, r: int, num_moe: int) -> float:
    """expert 하나의 scaling. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U model/peft/tuners/moelora.py:618``: ``lora_alpha / (r // num_moe)`` (D8 expert_rank).
    """
    return lora_alpha / (r // num_moe)


def moelora_einsum(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, gate: torch.Tensor, scaling: float,
                   r: int, K: int, *, dropout: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
    """expert별 einsum과 gate 합 (LoRA 항). 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``U model/peft/tuners/moelora.py:750-783`` (``calculate_B`` and the LoRA part
    of ``Linear.forward``). ``x`` is ``[batch, seq, d_in]`` (:761), ``A`` = ``lora_A.weight``
    ``[r, d_in]``, ``B`` = ``lora_B.weight`` ``[d_out, r]``, ``gate`` = omega ``[batch, K]``
    (the shared router returns ``[batch, 1, K]``, ``U model/router/nlpr.py:80``; a 2-D
    ``gate`` gets that middle axis here), ``scaling`` as at :618, ``dropout`` = the layer's
    ``lora_dropout`` (:775, identity when None). Returns ``(B_out * Gate).sum(dim=-2) * scaling``,
    the term that :783 adds in place to the base output ``F.linear(x, W, b)`` (:771).
    """
    if A.shape[0] != r:
        raise ValueError(f"A has {A.shape[0]} rows, expected r = {r}")
    batch_size, seq_len, _ = x.size()                                                     # :761
    x = x.to(A.dtype)                                                                     # :772
    if dropout is not None:
        x = dropout(x)                                                                    # :775
    A_out = F.linear(x, A).reshape(batch_size, seq_len, K, -1)                            # :774-776
    # calculate_B(A_out)                                                                  # :777
    _, _, n, r_star = A_out.size()                                                        # :751
    weight = B.t().reshape(n, r_star, -1)                                                 # :752
    B_out = torch.einsum('ijkl, klm->ijkm', A_out, weight)                                # :753
    gate_weights = gate if gate.dim() == 3 else gate.unsqueeze(1)                         # nlpr.py:80
    Gate = gate_weights.unsqueeze(-1)                                                     # :780
    return (B_out * Gate).sum(dim=-2) * scaling                                           # :783


# ----------------------------------------------------------------------------
# Fork sample numbering (M1 #8, measurement cross-check)
# ----------------------------------------------------------------------------
def fork_sample_idx(df: Any) -> dict[int, int]:
    """fork 표본 번호 = 필터 후 위치. 대조 전용, 실행 경로에서 쓰지 않음.

    Source: ``F data/movielens_data.py:30`` (``'sample_idx': i``) with ``:26``
    (``session_data.iloc[i]``) and ``:79-80`` (``train_data[train_data['len_seq'] >= 3]``).
    ``df`` is the released DataFrame as read from the ``.df`` file (row labels =
    released rows). Returns ``{released row label: sample_idx}`` in sample order.
    """
    session_data = df[df['len_seq'] >= 3]                                                 # :80
    return {int(label): i for i, label in enumerate(session_data.index)}                  # :26, :30


__all__ = [
    "first_line_and_hr1", "code_calculate_hr1", "code_prompt", "code_pair_input_and_labels", "wrap_emb_loop",
    "upstream_lr", "upstream_group_lr_lists", "upstream_scaling", "moelora_einsum", "fork_sample_idx",
]

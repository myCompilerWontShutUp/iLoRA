"""M14 RUNBOOK_datasets (DESIGN.md §23.6, §23.8 #14; D43, D44): ``rebuild/RUNBOOK_datasets.md`` against the real CLIs.

The bash blocks and the inline ``python -I rebuild/...`` commands are split with the small reader of
tests/test_m14_runbook.py, repeated here because test modules cannot import each other (test_m13_coverage
``test_no_repo_imports``). Every ``rebuild/main.py`` call parses with main.py's own parser and loads its ``--config`` (a
file of ``rebuild/configs/datasets/``) and ``--set`` values with M0 ``load_config``; ``compare_runs.py`` and
``convert_sasrec.py`` calls parse with their parsers; ``latest`` and ``rec`` follow the command and the dataset of their
step (``ILORA_RUN_<DATASET>_<STEP>``); the steps keep the order of §23.6 (DS0-DS2 with DS1 of both datasets before DS2,
LastFM DS3-DS9, Steam DS3-DS9, then the optional DS10-DS14); the DS7 pair has the form of RUNBOOK R8 (the second command
with ``eval.batch_invariance_check=true`` and the ``_BI`` record name); the bf16 alternative uses the dataset profiles
and never the MovieLens block of RUNBOOK.md; ``dsoutputs`` selects tests of tests/vessl/test_ds_run_outputs.py; no bash
line deletes or force-pushes and no credential appears. §23.8: no ``criterion`` marker in this module.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from ilora import commands, run
from ilora.config import ConfigError, load_config

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
PROFILE_DIR = REBUILD_DIR / "configs" / "datasets"
RUNBOOK = REBUILD_DIR / "RUNBOOK_datasets.md"
MAIN_RUNBOOK = REBUILD_DIR / "RUNBOOK.md"
RUN_OUTPUTS = REBUILD_DIR / "tests" / "vessl" / "test_ds_run_outputs.py"
DATASETS = ("lastfm", "steam")                                   # the order of §23.6 (LastFM first)
TITLES = {"lastfm": "LastFM", "steam": "Steam"}
HELPER_RE = re.compile(r"\s*(\w+)\(\)\s*\{(.*)\}\s*")            # a DS0/R0 helper definition such as rec() { ... }
STEP_RE = re.compile(r"(DS\d+)(?: \((LastFM|Steam)\))?\. ")      # "## DS3 (LastFM). ..." -> ("DS3", "LastFM")
SHELL_VARS = {"LLM_PATH", "ILORA_TEST_TOKENIZER", "HOME", "CONDA_PREFIX"}   # set outside the blocks or in DS0
REC_NAME_RE = re.compile(r"ILORA_RUN_(LASTFM|STEAM)_([A-Z0-9_]+)")
SECRET_NAME_RE = re.compile(r"(^|_)(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(_|$)", re.IGNORECASE)
BF16_SETS = ("model.torch_dtype=bfloat16", "train.precision=bf16_amp")   # §23.6 bf16 alternative, D9
BF16 = tuple(s.split("=", 1)[0] for s in BF16_SETS)
GENERIC_KEYWORDS = ("test_ds_run_outputs.py", "vessl", "tests", "rebuild", "skip", "slow", "pytestmark", "parametrize",
                    "usefixtures")                               # names pytest -k also matches (module, markers)
# §23.6: (kind, profile after "<d>_", --set keys) of every rebuild/main.py or compare_runs.py call of a step, document
# order; kind "resume" is ``train --resume`` (the profile is that of its ``latest``), kind "compare" is compare_runs.py.
CALLS: dict[str, list[tuple[str, Optional[str], tuple[str, ...]]]] = {
    "DS3": [("check-env", "paper_spec", ())],
    "DS4": [("smoke", "smoke_gpu_100", ())],
    "DS5": [("smoke", "overfit_gpu", ())],
    "DS6": [("train", "paper_spec", ()), ("resume", "paper_spec", ())],
    "DS7": [("eval", "paper_spec", ("eval.source_run",)),
            ("eval", "paper_spec", ("eval.source_run", "eval.batch_invariance_check"))],
    "DS8": [("train", "llara_baseline", ()), ("resume", "llara_baseline", ())],
    "DS9": [("compare", None, ())],
    "DS10": [("eval", "variant_text_only_eval", ("eval.source_run",))],
    "DS11": [("export", "paper_spec", ("records.source_run",))],
    "DS12": [("cluster", "paper_spec", ("cluster.dynamic_export_dir",))],
    "DS13": [("train", "cluster_hard", ("cluster.artifact_dir",)),
             ("export", "cluster_hard", ("cluster.artifact_dir", "records.source_run")),
             ("resume", "cluster_hard", ())],
    "DS14": [("train", "variant_released_lr", ()), ("compare", None, ())],
}
RECS = {"DS3": ["CHECK_ENV"], "DS4": ["SMOKE_GPU"], "DS5": ["OVERFIT"], "DS6": ["E2", "E2"],
        "DS7": ["E2_REEVAL", "E2_REEVAL_BI"], "DS8": ["E1", "E1"], "DS9": ["COMPARE"], "DS10": ["E3"],
        "DS11": ["E2_EXPORT", "E2_BEST"], "DS12": ["CLUSTER"], "DS13": ["E4", "E4_EXPORT", "E4"],
        "DS14": ["V_RELEASED_LR", "COMPARE_ALL"]}
OUTPUTS = {"DS3": [], "DS4": ["smoke_gpu"], "DS5": ["overfit"], "DS6": ["e2_run", "e2_run"], "DS7": ["e2_reeval"],
           "DS8": ["e1", "e1"], "DS9": [], "DS10": [], "DS11": ["e2_export"], "DS12": ["cluster"], "DS13": ["e4"],
           "DS14": []}
SMOKE_TRAIN = {"lastfm": 3_626, "steam": 1_385}                  # §23.6 train samples of <d>_smoke_gpu_100


def load_tool(path: Path, name: str) -> Any:
    """A script of rebuild/ imported from its file under a private name (for its ``build_parser``)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def parsers() -> dict[str, argparse.ArgumentParser]:
    return {"main": load_tool(REBUILD_DIR / "main.py", "ds_rebuild_main_under_test").build_parser(),
            "compare": load_tool(REBUILD_DIR / "tools" / "compare_runs.py", "ds_compare_runs_under_test")
            .build_parser(),
            "convert": load_tool(REBUILD_DIR / "tools" / "convert_sasrec.py", "ds_convert_sasrec_under_test")
            .build_parser()}


# ----------------------------------------------------------------------------
# A small reader of the runbook's shell lines (tests/test_m14_runbook.py; test modules do not import each other)
# ----------------------------------------------------------------------------
@dataclass
class Command:
    """One simple command of a bash block (``argv`` after the env assignments)."""
    argv: list[str]
    line: str
    block: int
    assigns: list[str] = field(default_factory=list)   # NAME of NAME=... tokens before the command
    in_function: bool = False                            # inside a DS0 helper definition such as outputs() { ... }


def bash_blocks(text: str) -> list[str]:
    return re.findall(r"^```bash\n(.*?)^```", text, flags=re.S | re.M)


def inline_commands(text: str) -> list[str]:
    """Inline code spans outside the bash blocks that call ``python -I rebuild/...`` (the resume commands)."""
    outside = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    return [span for span in re.findall(r"`([^`\n]+)`", outside) if "python -I rebuild/" in span]


def logical_lines(block: str) -> list[str]:
    return [line for line in block.replace("\\\n", " ").splitlines() if line.strip()]


def split_commands(line: str) -> list[list[str]]:
    """``shlex`` tokens (comments dropped) split at the shell operators ``&& || ; | ( ) > >>``."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    out: list[list[str]] = [[]]
    for token in lexer:
        if token and set(token) <= set("();<>|&"):
            out.append([])
        else:
            out[-1].append(token)
    return [c for c in out if c]


def read_runbook(text: str) -> tuple[list[Command], set[str]]:
    """(commands in document order, blocks first and then inline spans; names of the shell functions defined)."""
    sources = [(i, block) for i, block in enumerate(bash_blocks(text))]
    sources += [(len(sources) + i, span) for i, span in enumerate(inline_commands(text))]
    functions: set[str] = set()
    out: list[Command] = []
    for index, block in sources:
        for line in logical_lines(block):
            match = HELPER_RE.fullmatch(line)
            if match:                                   # DS0 helper: keep its body's commands for the path checks
                functions.add(match.group(1))
                line = match.group(2)
            for tokens in split_commands(line):
                assigns = []
                while tokens and re.match(r"[A-Za-z_]\w*=", tokens[0]):
                    assigns.append(tokens[0].split("=", 1)[0])
                    tokens = tokens[1:]
                out.append(Command(list(tokens), line, index, assigns, match is not None))
    return out, functions


def python_target(argv: list[str]) -> tuple[Optional[str], Optional[str], list[str]]:
    """``python [-I] (-m module | -c code | script) args`` -> (module, script, args)."""
    if not argv or argv[0] not in ("python", "python3"):
        return None, None, []
    i = 1
    while i < len(argv) and argv[i] in ("-I", "-u", "-q", "-B", "-E", "-s"):
        i += 1
    if i < len(argv) and argv[i] == "-m":
        return argv[i + 1], None, argv[i + 2:]
    if i < len(argv) and argv[i] == "-c":
        return None, None, argv[i + 2:]
    return (None, argv[i], argv[i + 1:]) if i < len(argv) else (None, None, [])


def parse(parser: argparse.ArgumentParser, args: list[str], where: str) -> argparse.Namespace:
    """``parser.parse_args`` that turns argparse's exit into a test failure with its message."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return parser.parse_args(args)
    except SystemExit:
        pytest.fail(f"{where}: rejected by {parser.prog}: {err.getvalue().strip()}")


def step_sections(text: str) -> list[tuple[str, str]]:
    """(key, text) of every ``## `` section in document order: ``DS3/lastfm`` or ``DS0`` for a step, else its title."""
    out = []
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        title = part.splitlines()[0]
        match = STEP_RE.match(title)
        if match:
            key = match.group(1) if match.group(2) is None else f"{match.group(1)}/{match.group(2).lower()}"
        else:
            key = title.strip()
        out.append((key, part))
    return out


def section_sources(section: str) -> list[str]:
    """The bash blocks and the inline ``python -I rebuild/...`` spans of one section in document order."""
    found = [(m.start(), m.group(1)) for m in re.finditer(r"^```bash\n(.*?)^```", section, flags=re.S | re.M)]
    outside = re.sub(r"^```.*?^```", lambda m: " " * len(m.group(0)), section, flags=re.S | re.M)
    found += [(m.start(), m.group(1)) for m in re.finditer(r"`([^`\n]+)`", outside)
              if "python -I rebuild/" in m.group(1)]
    return [source for _, source in sorted(found)]


def chains(line: str) -> list[list[list[str]]]:
    """The ``&&`` chains of one logical line, each a list of commands (``shlex`` tokens). ``;``, ``||``, ``|``, ``&``
    and redirections end a chain; the parentheses of ``$(...)`` end a command only (a capture stays in its chain)."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    out: list[list[list[str]]] = [[]]
    command: list[str] = []
    for token in lexer:
        if token and set(token) <= set("();<>|&"):
            if command:
                out[-1].append(command)
                command = []
            if token.replace("(", "").replace(")", "") not in ("", "&&"):
                out.append([])
        else:
            command.append(token)
    if command:
        out[-1].append(command)
    return [chain for chain in out if chain]


def chain_commands(chain: list[list[str]]) -> list[tuple[list[str], list[str]]]:
    """(assigned names, argv) of each command of a chain."""
    out = []
    for tokens in chain:
        assigns = []
        while tokens and re.match(r"[A-Za-z_]\w*=", tokens[0]):
            assigns.append(tokens[0].split("=", 1)[0])
            tokens = tokens[1:]
        out.append((assigns, tokens))
    return out


def section_chains(section: str) -> list[list[tuple[list[str], list[str]]]]:
    """Each ``&&`` chain (list of (assigned names, argv)) of a section in document order, DS0 helper bodies left out."""
    out = []
    for source in section_sources(section):
        for line in logical_lines(source):
            if not HELPER_RE.fullmatch(line):
                out.extend(chain_commands(chain) for chain in chains(line))
    return out


def sections_of(text: Optional[str] = None) -> dict[str, str]:
    return dict(step_sections(RUNBOOK.read_text(encoding="utf-8") if text is None else text))


def main_runbook_section(step: str) -> str:
    """The ``## R<n>. ...`` section of RUNBOOK.md (the MovieLens runbook the dataset runbook follows)."""
    (section,) = [part for part in re.split(r"^## ", MAIN_RUNBOOK.read_text(encoding="utf-8"), flags=re.M)[1:]
                  if part.startswith(f"{step}. ")]
    return section


def resolve_call(argv: list[str], latest: Optional[str], main_parser: argparse.ArgumentParser
                 ) -> Optional[tuple[str, Optional[str], tuple[str, ...]]]:
    """``(kind, profile stem, --set keys)`` of one command if it is a ``main.py`` or ``compare_runs.py`` call, else None
    (``resume``: the profile stem is the one of ``latest`` before it)."""
    _, script, args = python_target(argv)
    if script == "rebuild/tools/compare_runs.py":
        return "compare", None, ()
    if script != "rebuild/main.py":
        return None
    ns = parse(main_parser, args, " ".join(argv))
    if ns.resume is not None:
        return "resume", latest, ()
    return ns.command, Path(ns.config).stem if ns.config else None, tuple(s.split("=", 1)[0] for s in ns.sets)


def calls_of(section: str, main_parser: argparse.ArgumentParser) -> list[tuple[str, Optional[str], tuple[str, ...]]]:
    """Every main.py or compare_runs.py call of a section in document order (the resume spans included)."""
    out = []
    for chain in section_chains(section):
        latest: Optional[str] = None
        for _, argv in chain:
            if argv[:1] == ["latest"]:
                latest = argv[1]
            call = resolve_call(argv, latest, main_parser)
            if call is not None:
                out.append(call)
    return out


def command_names(section: str, head: str) -> list[str]:
    """The first argument of every ``<head> ...`` command of a section in document order (``rec``, ``dsoutputs``)."""
    return [argv[1] for chain in section_chains(section) for _, argv in chain if argv[:1] == [head]]


def run_output_steps() -> tuple[dict[str, list[str]], dict[str, tuple[str, ...]], tuple[str, ...]]:
    """(step -> test functions, ``STEP_RUNS``, ``DATASETS``) of tests/vessl/test_ds_run_outputs.py, read with ast."""
    tree = ast.parse(RUN_OUTPUTS.read_text(encoding="utf-8"))
    literals: dict[str, Any] = {}
    steps: dict[str, list[str]] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("STEP_RUNS", "DATASETS")):
            literals[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names = [d.args[0].value for d in node.decorator_list if isinstance(d, ast.Call)
                     and isinstance(d.func, ast.Name) and d.func.id == "step"]
            assert len(names) == 1, f"{node.name}: no @step(...) decorator"
            steps.setdefault(names[0], []).append(node.name)
    return steps, literals["STEP_RUNS"], literals["DATASETS"]


# ----------------------------------------------------------------------------
# §23.8 #14: order, commands, names
# ----------------------------------------------------------------------------
def test_steps_come_in_the_design_order():
    """§23.6 실행 순서: DS0, DS1 of both datasets, DS2, LastFM DS3-DS9, Steam DS3-DS9, then the optional DS10-DS14 of each
    dataset, and the two rules (run order, fail-stop) are written in the document."""
    text = RUNBOOK.read_text(encoding="utf-8")
    keys = [key for key, _ in step_sections(text) if re.fullmatch(r"DS\d+(/\w+)?", key)]
    required = ["DS0", "DS1", "DS2", *[f"DS{i}/lastfm" for i in range(3, 10)], *[f"DS{i}/steam" for i in range(3, 10)]]
    optional = [f"DS{i}/{d}" for d in DATASETS for i in range(10, 15)]
    assert keys == required + optional
    sections = sections_of(text)
    rules, order = sections["규칙"], sections["실행 순서와 실패 시 멈춤"]
    assert "구현 commit은 54e5ae0의 MovieLens 실행이 모두 끝난 뒤(또는 사용자가 중단한 뒤)에만" in rules
    for phrase in ("DS1을 LastFM과 Steam 둘 다", "DS2(시험 묶음)", "LastFM DS3–DS9", "(T 1,870)",
                   "Steam DS3–DS9는 LastFM DS3–DS6이 통과한 뒤 시작한다",
                   "어느 데이터셋의 DS1 또는 DS2가 실패하면 DS3 이후를 시작하지 않는다",
                   "LastFM DS3·DS4·DS5가 실패하면 Steam DS3 이후를 시작하지 않는다",
                   "Steam 진행은 사용자가 정한다", "GradScaler skip 연속 16회"):
        assert phrase in order, phrase
    assert "LastFM DS3–DS6이 통과한 뒤에 시작한다" in sections["DS3/steam"]


def test_ds0_is_r0_plus_dsoutputs():
    """§23.6 DS0: the block of RUNBOOK.md R0 line by line (the same helpers, ``source`` of the same paths file), plus
    only the new ``dsoutputs`` helper that runs ``tests/vessl/test_ds_run_outputs.py`` with ``-k``."""
    sections = sections_of()
    ds0 = bash_blocks(sections["DS0"])
    (r0,) = bash_blocks(main_runbook_section("R0"))
    assert len(ds0) == 1
    r0_lines, ds0_lines = r0.splitlines(), ds0[0].splitlines()
    assert ds0_lines == [*r0_lines, ('dsoutputs() { ILORA_VESSL=1 python -I -m pytest '
                                     'rebuild/tests/vessl/test_ds_run_outputs.py -q -k "$1"; }')]


def test_commands_are_valid(parsers):
    """Every ``rebuild/main.py`` call parses with main.py's own parser, names one of the six commands and a profile of
    ``configs/datasets/`` whose ``--set`` values load (M0 ``load_config``); ``compare_runs.py`` and
    ``convert_sasrec.py`` calls parse with their parsers; pytest paths exist; ``latest <name>`` names a dataset profile
    or ``compare``; shell variables are set before a ``rebuild/`` script reads them."""
    text = RUNBOOK.read_text(encoding="utf-8")
    cmds, functions = read_runbook(text)
    assert {"latest", "done_ok", "rec", "need", "outputs", "dsoutputs"} <= functions
    profiles = {p.stem for p in PROFILE_DIR.glob("*.yaml")}
    assert len(profiles) == 18
    assigned = set(SHELL_VARS)
    counts = dict.fromkeys(("main", "compare", "convert", "pytest", "latest"), 0)
    for cmd in cmds:
        argv = cmd.argv
        where = f"RUNBOOK_datasets: {cmd.line.strip()}"
        module, script, args = python_target(argv)
        refs = sorted({m for a in args for m in re.findall(r"\$\{?([A-Za-z_]\w*)", a)} - assigned)
        if script is not None and script.startswith("rebuild/"):
            assert (REPO_ROOT / script).is_file(), where
            assert not refs, f"{where}: {refs} used before they are set"
        if script == "rebuild/main.py":
            counts["main"] += 1
            ns = parse(parsers["main"], args, where)
            assert ns.command in commands.COMMANDS, where
            if ns.resume is not None:                    # train --resume <folder>: no --config/--set (M10 14.6.10)
                assert ns.command == "train" and ns.config is None and not ns.sets, where
            else:
                config = REPO_ROOT / ns.config if ns.config is not None else None
                assert config is not None and config.is_file() and config.parent == PROFILE_DIR, where
                assert config.stem in profiles, where
                try:
                    cfg, applied = load_config(config, ns.sets)
                    commands.command_requirements(ns.command, cfg)
                except ConfigError as exc:
                    pytest.fail(f"{where}: {exc}")
                assert len(applied) == len(ns.sets), where
        elif script == "rebuild/tools/compare_runs.py":
            counts["compare"] += 1
            ns = parse(parsers["compare"], args, where)
            names = [name for name, _ in ns.runs]
            assert len(names) == len(set(names)) and {"E1", "E2"} <= set(names), where
            assert {name for name, _ in ns.exports} <= set(names) and ns.root is None, where
        elif script == "rebuild/tools/convert_sasrec.py":
            counts["convert"] += 1
            ns = parse(parsers["convert"], args, where)
            assert ns.repo == Path(".") and ns.dataset in DATASETS, where
            cfg, _ = load_config(PROFILE_DIR / f"{ns.dataset}_paper_spec.yaml")
            expected_out = Path(f"runs/rebuild/assets/sasrec_{ns.dataset}")
            assert ns.out == Path(cfg.sasrec.state_path).parent == expected_out, where
        elif module == "pytest":
            paths = [a for a in args if not a.startswith("-") and a.startswith("rebuild/")]
            assert paths and all((REPO_ROOT / p).exists() for p in paths), where
            counts["pytest"] += 0 if cmd.in_function else 1
        elif argv[:1] == ["latest"]:
            counts["latest"] += 1
            assert argv[1] in profiles or argv[1] == "compare", f"{where}: latest {argv[1]} is not a dataset profile"
        assigned.update(cmd.assigns)
        if argv[:1] == ["rec"]:
            assigned.add(argv[1])
        if argv[:1] == ["export"]:
            assigned.update(a.split("=", 1)[0] for a in argv[1:] if "=" in a)
    assert counts["main"] >= 44 and counts["compare"] == 4 and counts["convert"] == 2 and counts["pytest"] == 2
    assert counts["latest"] >= 44                  # one after every folder-making command and before every resume


def test_ds1_precedes_ds2(parsers):
    """§23.6: both DS1 conversions (LastFM, then Steam) come before the DS2 test bundle, with ``--dataset`` and the
    output folder of the profile, and DS2 is the pair of commands of RUNBOOK.md R3."""
    sections = sections_of()
    conversions = [cmd for cmd in read_runbook(sections["DS1"])[0]
                   if python_target(cmd.argv)[1] == "rebuild/tools/convert_sasrec.py"]
    assert [parse(parsers["convert"], python_target(c.argv)[2], c.line).dataset for c in conversions] == list(DATASETS)
    for cmd, name in zip(conversions, DATASETS):
        assert cmd.argv[cmd.argv.index("--out") + 1] == f"runs/rebuild/assets/sasrec_{name}", cmd.line
    text = RUNBOOK.read_text(encoding="utf-8")
    assert text.index("## DS1. ") < text.index("## DS2. ")
    (ds2,) = bash_blocks(sections["DS2"])
    (r3,) = bash_blocks(main_runbook_section("R3"))
    assert split_commands(ds2) == split_commands(r3)                     # the two pytest commands, tokens compared
    all_cmds = read_runbook(text)[0]
    pytest_at = [i for i, c in enumerate(all_cmds) if python_target(c.argv)[0] == "pytest" and not c.in_function]
    convert_at = [i for i, c in enumerate(all_cmds) if python_target(c.argv)[1] == "rebuild/tools/convert_sasrec.py"]
    assert len(convert_at) == 2 and len(pytest_at) == 2 and max(convert_at) < min(pytest_at)


@pytest.mark.parametrize("dataset", DATASETS)
def test_calls_of_every_step(parsers, dataset):
    """§23.6: the (command, profile, ``--set`` keys) of every step of the dataset, the resume spans included, in order;
    the profile names carry the dataset (D43)."""
    sections = sections_of()
    for step, expected in CALLS.items():
        key = f"{step}/{dataset}"
        want = [(kind, None if name is None else f"{dataset}_{name}", sets) for kind, name, sets in expected]
        got = calls_of(sections[key], parsers["main"])
        assert got == want, key


@pytest.mark.parametrize("dataset", DATASETS)
def test_comparison_calls(parsers, dataset):
    """DS9 compares E1 and E2 (§23.6); DS14 repeats it with every optional run, and every ILORA name of both commands is
    a name of this dataset, checked with ``need`` in the same chain (R14, R15)."""
    upper = dataset.upper()
    sections = sections_of()
    (ds9,) = [argv for chain in section_chains(sections[f"DS9/{dataset}"]) for _, argv in chain
              if python_target(argv)[1] == "rebuild/tools/compare_runs.py"]
    ds14 = [argv for chain in section_chains(sections[f"DS14/{dataset}"]) for _, argv in chain
            if python_target(argv)[1] == "rebuild/tools/compare_runs.py"]
    assert len(ds14) == 1
    first = parse(parsers["compare"], python_target(ds9)[2], "DS9")
    assert [name for name, _ in first.runs] == ["E1", "E2"] and first.exports == [] and first.cluster is None
    last = parse(parsers["compare"], python_target(ds14[0])[2], "DS14")
    assert [name for name, _ in last.runs] == ["E1", "E2", "E3", "E4", "V_released_lr"]
    assert [name for name, _ in last.exports] == ["E2", "E4"] and last.cluster == f"$ILORA_RUN_{upper}_CLUSTER"
    for argv in (ds9, ds14[0]):
        refs = {m for a in argv for m in re.findall(r"\$\{?([A-Za-z_]\w*)", a)}
        assert refs and all(r.startswith(f"ILORA_RUN_{upper}_") for r in refs), argv


@pytest.mark.parametrize("dataset", DATASETS)
def test_rec_names_match_the_dataset_and_step(dataset):
    """Each ``rec`` of a step records ``ILORA_RUN_<DATASET>_<STEP>`` of its dataset (§23.6), so nothing overwrites the
    MovieLens names of RUNBOOK.md (``ILORA_RUN_E2`` etc.); DS6 and DS8 record again in the resume spans."""
    sections = sections_of()
    for step, names in RECS.items():
        key = f"{step}/{dataset}"
        got = command_names(sections[key], "rec")
        assert got == [f"ILORA_RUN_{dataset.upper()}_{name}" for name in names], key
        for rec in got:
            match = REC_NAME_RE.fullmatch(rec)
            assert match and match.group(1) == dataset.upper(), rec


def test_every_recorded_name_is_a_dataset_name():
    """No ``rec`` in the document writes a name outside ``ILORA_RUN_<DATASET>_<STEP>`` (the bf16 blocks included)."""
    for cmd in read_runbook(RUNBOOK.read_text(encoding="utf-8"))[0]:
        if cmd.argv[:1] == ["rec"]:
            assert REC_NAME_RE.fullmatch(cmd.argv[1]), cmd.line


@pytest.mark.parametrize("dataset", DATASETS)
def test_ds7_pair_has_the_form_of_r8(dataset):
    """§23.6 DS7: the block is RUNBOOK.md R8 with the dataset's profile and names: two ``eval`` runs of the E2
    checkpoint, the second with ``--set eval.batch_invariance_check=true`` and the record name
    ``ILORA_RUN_<D>_E2_REEVAL_BI``."""
    upper = dataset.upper()
    sections = sections_of()
    (ds7,) = bash_blocks(sections[f"DS7/{dataset}"])
    (r8,) = bash_blocks(main_runbook_section("R8"))
    expected = (r8.replace("rebuild/configs/paper_spec.yaml", f"rebuild/configs/datasets/{dataset}_paper_spec.yaml")
                .replace("ILORA_RUN_E2", f"ILORA_RUN_{upper}_E2")
                .replace("latest paper_spec", f"latest {dataset}_paper_spec")
                .replace("outputs e2_reeval", f"dsoutputs {dataset}_e2_reeval"))
    assert ds7 == expected
    lines = logical_lines(ds7)
    assert len(lines) == 2
    assert "--set eval.batch_invariance_check=true" not in lines[0]
    assert f"ILORA_RUN_{upper}_E2_REEVAL_BI" not in lines[0]
    assert "--set eval.batch_invariance_check=true" in lines[1] and f"rec ILORA_RUN_{upper}_E2_REEVAL_BI" in lines[1]
    assert f"dsoutputs {dataset}_e2_reeval" in lines[1]


@pytest.mark.parametrize("dataset", DATASETS)
def test_bf16_alternative(parsers, dataset):
    """§23.6 bf16 alternative (D9): this dataset's block has the commands of DS6 and DS7 with
    ``<d>_variant_bf16.yaml`` and of DS8, DS10, DS13 and DS14 with the two ``--set`` values; every run it starts loads
    as bf16; it records only ``ILORA_RUN_<D>_...`` and names no MovieLens profile or ``ILORA_RUN_E2``."""
    upper = dataset.upper()
    sections = sections_of()
    (title,) = [key for key in sections if key.startswith("bf16 대안")]
    section = sections[title]
    assert "R7의 실패 규칙" in section and "편차로 기록" in section and "RUNBOOK.md" in section
    assert '"bf16 대안" 절의 블록은 쓰지 않는다' in section and "MovieLens를 다시 학습" in section
    profile = f"--config rebuild/configs/datasets/{dataset}_variant_bf16.yaml"
    (block,) = [b for b in bash_blocks(section) if profile in b]
    runs = []
    for line in logical_lines(block):
        for chain in chains(line):
            for _, argv in chain_commands(chain):
                module, script, args = python_target(argv)
                if script != "rebuild/main.py":
                    continue
                ns = parse(parsers["main"], args, line)
                cfg, _ = load_config(REPO_ROOT / ns.config, ns.sets)
                assert (cfg.model.torch_dtype, cfg.train.precision) == ("bfloat16", "bf16_amp"), line
                assert ns.config.parent == Path("rebuild/configs/datasets") and ns.config.stem.startswith(f"{dataset}_")
                bf16_sets = [s for s in ns.sets if s in BF16_SETS]
                if ns.config.stem == f"{dataset}_variant_bf16":
                    assert not bf16_sets, line                   # the profile already holds them
                else:
                    assert bf16_sets == list(BF16_SETS), line
                runs.append((ns.command, ns.config.stem))
    assert runs == [("train", f"{dataset}_variant_bf16"), ("eval", f"{dataset}_variant_bf16"),
                    ("eval", f"{dataset}_variant_bf16"), ("train", f"{dataset}_llara_baseline"),
                    ("eval", f"{dataset}_variant_text_only_eval"), ("train", f"{dataset}_cluster_hard"),
                    ("train", f"{dataset}_variant_released_lr")]
    recs = [argv[1] for line in logical_lines(block) for chain in chains(line) for _, argv in chain_commands(chain)
            if argv[:1] == ["rec"]]
    assert recs == [f"ILORA_RUN_{upper}_{name}" for name in ("E2", "E2_REEVAL", "E2_REEVAL_BI", "E1", "E3", "E4",
                                                              "V_RELEASED_LR")]
    assert "ILORA_RUN_E2" not in block.replace(f"ILORA_RUN_{upper}_E2", "")
    assert "rebuild/configs/variant_bf16.yaml" not in block
    assert f"latest {dataset}_variant_bf16" in block


# ----------------------------------------------------------------------------
# §23.6 rules (a new shell, folders, selectors, credentials, deletions) and the counts of the text
# ----------------------------------------------------------------------------
def test_paths_survive_a_new_shell():
    """RUNBOOK.md rule (paths restored by ``source`` after a lost shell): every ``$NAME`` that a ``rebuild/main.py`` or
    ``compare_runs.py`` call reads is set earlier in the same step, by DS0, or is an ``ILORA_*`` name that ``rec``
    recorded in an earlier step; every such ``ILORA_*`` name is checked with ``need`` earlier in the same ``&&``
    chain."""
    restored: set[str] = set()                                  # rec names of earlier sections (runbook_paths.sh)
    ds0: set[str] = set(SHELL_VARS)
    assigned: dict[str, set[str]] = {}
    checked = []
    for key, section in step_sections(RUNBOOK.read_text(encoding="utf-8")):
        here = assigned.setdefault(key, set())
        for chain in section_chains(section):
            needed: set[str] = set()
            for names, argv in chain:
                if argv[:1] == ["need"]:
                    needed.update(argv[1:])
                _, script, args = python_target(argv)
                if script in ("rebuild/main.py", "rebuild/tools/compare_runs.py"):
                    refs = {m for a in args for m in re.findall(r"\$\{?([A-Za-z_]\w*)", a)}
                    where = f"{key}: {' '.join(argv)[:120]}"
                    missing = refs - here - restored - ds0
                    assert not missing, f"{where}: {sorted(missing)} lost in a new shell"
                    unchecked = {r for r in refs if r.startswith("ILORA_")} - needed
                    assert not unchecked, f"{where}: {sorted(unchecked)} not checked with need first"
                    checked.append(key)
                here.update(names)
                if argv[:1] == ["rec"]:
                    here.add(argv[1])
                if argv[:1] == ["export"]:
                    here.update(a.split("=", 1)[0] for a in argv[1:] if "=" in a)
        if key == "DS0":
            ds0 |= here
        else:
            restored |= {n for n in here if n.startswith("ILORA_")}
    assert {f"DS{i}/{d}" for i in (7, 9, 10, 11, 12, 13, 14) for d in DATASETS} <= set(checked)
    assert [k for k in checked if k.startswith("bf16")], "the bf16 blocks were not read"


def test_folder_captures_follow_their_command(parsers):
    """``X=$(latest <profile>)`` takes the folder of the command just before it in the same ``&&`` chain (so a command
    that fails before making its folder stops the chain), or the folder that ``train --resume`` reopens right after
    it; the name is the profile of that command (or ``compare``) and the following ``done_ok`` names the command
    (RUNBOOK.md)."""
    seen = 0
    for key, section in step_sections(RUNBOOK.read_text(encoding="utf-8")):
        for chain in section_chains(section):
            commands_ = [(names, argv) for names, argv in chain if argv]
            for i, (_, argv) in enumerate(commands_):
                if argv[:1] == ["done_ok"]:
                    assert len(argv) == 3 and argv[2] in (*commands.COMMANDS, "compare"), f"{key}: {argv}"
                if argv[:1] != ["latest"]:
                    continue
                before = resolve_call(commands_[i - 1][1], None, parsers["main"]) if i else None
                after = resolve_call(commands_[i + 1][1], None, parsers["main"]) if i + 1 < len(commands_) else None
                if before is not None and before[0] != "resume":
                    maker, stem = before[0], before[1] if before[0] != "compare" else "compare"
                else:
                    where = f"{key}: latest {argv[1]} does not follow its command"
                    assert after is not None and after[0] == "resume", where
                    maker, stem = "train", argv[1]
                assert argv[1] == stem, f"{key}: latest {argv[1]} is not the profile of its command ({stem})"
                checks = [a for _, a in commands_[i + 1:] if a[:1] == ["done_ok"]]
                assert checks and checks[0][2] == maker, f"{key}: latest {argv[1]} is not checked as a {maker} folder"
                seen += 1
    assert seen >= 48


def test_output_selectors():
    """Each ``dsoutputs <dataset>_<step>`` names a step of tests/vessl/test_ds_run_outputs.py (``-k`` selects the test
    ids ``<dataset>_<step>``) whose ``ILORA_RUN_<DATASET>_<NAME>`` folders were recorded with ``rec`` before it; every
    step of that file is selected for both datasets, and the steps of §23.6 use the selectors of its table."""
    text = RUNBOOK.read_text(encoding="utf-8")
    steps, step_runs, datasets = run_output_steps()
    assert tuple(datasets) == DATASETS and set(steps) == set(step_runs) and all(steps.values())
    recorded: set[str] = set()
    selected: dict[str, set[str]] = {d: set() for d in DATASETS}
    for cmd in read_runbook(text)[0]:
        if cmd.argv[:1] == ["rec"]:
            recorded.add(cmd.argv[1])
        elif cmd.argv[:1] == ["dsoutputs"]:
            selector = cmd.argv[1]
            dataset, _, step = selector.partition("_")
            assert dataset in DATASETS and step in steps, f"dsoutputs {selector} selects no test step"
            assert not [k for k in GENERIC_KEYWORDS if selector in k], f"dsoutputs {selector} selects everything"
            need = {f"ILORA_RUN_{dataset.upper()}_{name}" for name in step_runs[step]}
            assert need <= recorded, f"dsoutputs {selector}: {sorted(need - recorded)} not recorded yet"
            selected[dataset].add(step)
    assert all(selected[d] == set(step_runs) for d in DATASETS), selected
    sections = sections_of(text)
    for step, names in OUTPUTS.items():
        for dataset in DATASETS:
            key = f"{step}/{dataset}"
            assert command_names(sections[key], "dsoutputs") == [f"{dataset}_{name}" for name in names], key
    (bf16,) = [section for key, section in sections.items() if key.startswith("bf16")]
    assert command_names(bf16, "dsoutputs") == [f"{d}_{s}" for d in DATASETS
                                                for s in ("e2_run", "e2_reeval", "e1", "e2_run")]


def test_no_credentials():
    """No token-like value and no assignment to a credential variable [SPEC §0.9] (this document names no token)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    assert run.find_secrets([text]) == []
    for cmd in read_runbook(text)[0]:
        names = list(cmd.assigns)
        if cmd.argv[:1] == ["export"]:
            names += [a.split("=", 1)[0] for a in cmd.argv[1:]]
        elif cmd.argv[:1] == ["rec"]:
            names.append(cmd.argv[1])
        assert not [n for n in names if SECRET_NAME_RE.search(n)], cmd.line
    assert not re.search(r"HF_TOKEN\s*=", text)


def test_no_bash_line_deletes_or_force_pushes():
    """§23.6, RUNBOOK.md rules: the document has no delete command and no forced git operation; run folders are never
    overwritten or removed."""
    for cmd in read_runbook(RUNBOOK.read_text(encoding="utf-8"))[0]:
        argv = cmd.argv
        if not argv:
            continue
        assert argv[0] not in ("rm", "rmdir", "unlink", "shred"), cmd.line
        assert not (argv[0] == "find" and "-delete" in argv), cmd.line
        forbidden_git = (["clean"], ["reset"], ["push"], ["checkout"], ["stash"])
        assert not (argv[0] == "git" and argv[1:2] in forbidden_git), cmd.line


@pytest.mark.parametrize("dataset", DATASETS)
def test_counts_in_the_text(dataset):
    """§23.2, §23.6: the counts named by the checks of DS4, DS6, DS8, DS11, DS12 and DS13 equal M10 ``step_counts`` of
    the dataset's profiles and the sample counts of ``ilora.datasets``."""
    from ilora.datasets import SPECS
    from ilora.train import step_counts

    spec, sections = SPECS[dataset], sections_of()
    cfg, _ = load_config(PROFILE_DIR / f"{dataset}_paper_spec.yaml")
    sc = step_counts(spec.expected_samples["train"], cfg.train)
    train, micro, steps = f"{sc.n_samples:,}", f"{sc.total_micro:,}", f"{sc.total_steps:,}"
    val, test = f"{spec.expected_samples['val']:,}", f"{spec.expected_samples['test']:,}"
    ds6, ds8 = sections[f"DS6/{dataset}"], sections[f"DS8/{dataset}"]
    assert f"micro {micro}, step {steps}" in ds6 and f"validation {val}, test {test}" in ds6
    assert ("epoch 끝 창 없음" if sc.tail == 0 else f"epoch 끝 `n_micro` {sc.tail}") in ds6
    assert f"micro {micro}, step {steps}" in ds8
    assert f"train {train} · validation {val} · test {test}" in sections[f"DS11/{dataset}"]
    assert f"fit {train}행" in sections[f"DS12/{dataset}"]
    ds13 = sections[f"DS13/{dataset}"]
    assert f"{sc.total_micro * sc.micro_batch:,}" in ds13 and f"micro {micro} × {sc.micro_batch}" in ds13
    assert (sc.epochs, sc.tail == 0) == (5, dataset == "lastfm")
    # §23.6: the 2-epoch smoke of <d>_smoke_gpu_100 and its 1-epoch E4 path
    smoke_cfg, _ = load_config(PROFILE_DIR / f"{dataset}_smoke_gpu_100.yaml")
    smoke = step_counts(SMOKE_TRAIN[dataset], smoke_cfg.train)
    ds4 = sections[f"DS4/{dataset}"]
    assert f"train 표본 {SMOKE_TRAIN[dataset]:,}" in ds4
    assert f"micro {smoke.total_micro:,}, step {smoke.total_steps}" in ds4
    assert f"epoch 끝 `n_micro` {smoke.tail or smoke.accum}" in ds4
    assert f"`gate_calls` 합 {smoke.total_micro:,}" in ds4
    assert f"E4 경로 `cluster_lookups` {smoke.micro_per_epoch * smoke.micro_batch:,}" in ds4

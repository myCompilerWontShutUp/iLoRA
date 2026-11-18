"""M14 RUNBOOK과 계산된 학습 길이 (DESIGN.md §18.6.5, §18.6.6, §18.8 #4-#6).

``test_commands_are_valid`` splits every bash block of ``rebuild/RUNBOOK.md`` (and the inline commands that call
``python -I rebuild/...``) with ``shlex`` and checks each command against the real command lines: ``rebuild/main.py``
(its ``build_parser``, one of the six commands, the ``--config`` file, every ``--set`` key loaded by M0
``load_config`` with its type), ``rebuild/tools/compare_runs.py`` and ``rebuild/tools/convert_sasrec.py`` (their
``build_parser``), the pytest paths, and ``latest <name>`` = the profile of the command that made the folder (or
``compare``). ``test_output_selectors`` ties each ``outputs <selector>`` to the tests of
``tests/vessl/test_run_outputs.py`` and the ``ILORA_RUN_<NAME>`` folders they read. ``test_paths_survive_a_new_shell``
and ``test_folder_captures_follow_their_command`` read the steps one at a time, as a new shell after R0 would run them
(§18.6.6: paths come back from ``runbook_paths.sh``); further tests check the R1, R3, R7, R15 and bf16 rules.
``test_derived_lengths`` and ``test_overfit_rows`` recompute the §18.6.5 numbers with the M1, M4 and M10 functions.
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
CONFIG_DIR = REBUILD_DIR / "configs"
RUNBOOK = REBUILD_DIR / "RUNBOOK.md"
RUN_OUTPUTS = REBUILD_DIR / "tests" / "vessl" / "test_run_outputs.py"
STEPS = tuple(f"R{i}" for i in range(17))                       # §18.6.6 R0-R16
PROFILES = tuple(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")))
SHELL_VARS = {"LLM_PATH", "ILORA_TEST_TOKENIZER", "HOME", "CONDA_PREFIX"}   # set outside the blocks or in R0
# §18.7 order: tests -> check-env -> smoke 2 -> overfit -> E2 -> re-evaluation -> E3 -> export -> cluster -> E4 -> E1
# -> comparison -> variants. (command, profile) of the first main.py call of each step.
ORDER = (("check-env", "paper_spec"), ("smoke", "smoke_gpu_100"), ("smoke", "overfit_gpu"), ("train", "paper_spec"),
         ("eval", "paper_spec"), ("eval", "variant_text_only_eval"), ("export", "paper_spec"), ("cluster", "paper_spec"),
         ("train", "cluster_hard"), ("export", "cluster_hard"), ("train", "llara_baseline"),
         ("export", "llara_baseline"), ("compare", "compare"), ("train", "variant_all_hybrid"),
         ("train", "variant_user_level_cluster"))
LATE_STEPS = tuple(f"R{i}" for i in range(8, 17))              # steps that read the paths of earlier steps
HELPER_RE = re.compile(r"\s*(\w+)\(\)\s*\{(.*)\}\s*")            # an R0 helper definition such as rec() { ... }
BF16_SETS = ("model.torch_dtype=bfloat16", "train.precision=bf16_amp")   # §18.6.4 variant_bf16 keys
GENERIC_KEYWORDS = ("test_run_outputs.py", "vessl", "tests", "rebuild", "criterion", "skip", "slow", "pytestmark",
                    "parametrize", "usefixtures")              # names pytest -k also matches (module, markers)
SECRET_NAME_RE = re.compile(r"(^|_)(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(_|$)", re.IGNORECASE)


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
    return {"main": load_tool(REBUILD_DIR / "main.py", "rebuild_main_under_test").build_parser(),
            "compare": load_tool(REBUILD_DIR / "tools" / "compare_runs.py", "compare_runs_cli_under_test").build_parser(),
            "convert": load_tool(REBUILD_DIR / "tools" / "convert_sasrec.py", "convert_sasrec_cli_under_test")
            .build_parser()}


# ----------------------------------------------------------------------------
# A small reader of the RUNBOOK's shell lines
# ----------------------------------------------------------------------------
@dataclass
class Command:
    """One simple command of a bash block (``argv`` after the env assignments, ``$V``-style literals expanded)."""
    argv: list[str]
    line: str
    block: int
    assigns: list[str] = field(default_factory=list)   # NAME of NAME=... tokens before the command
    in_function: bool = False                            # inside an R0 helper definition such as outputs() { ... }


def bash_blocks(text: str) -> list[str]:
    return re.findall(r"^```bash\n(.*?)^```", text, flags=re.S | re.M)


def inline_commands(text: str) -> list[str]:
    """Inline code spans outside the bash blocks that call ``python -I rebuild/...`` (e.g. the resume commands)."""
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


def expand(token: str, variables: dict[str, str]) -> str:
    return re.sub(r"\$\{?(\w+)\}?", lambda m: variables.get(m.group(1), m.group(0)), token)


def read_runbook(text: str) -> tuple[list[Command], dict[str, str], set[str]]:
    """(commands in document order, literal variables, names of shell functions defined in the blocks)."""
    sources = [(i, block) for i, block in enumerate(bash_blocks(text))]
    sources += [(len(sources) + i, span) for i, span in enumerate(inline_commands(text))]
    variables: dict[str, str] = {}
    functions: set[str] = set()
    out: list[Command] = []
    for index, block in sources:
        for line in logical_lines(block):
            match = re.fullmatch(r"\s*(\w+)\(\)\s*\{(.*)\}\s*", line)
            if match:                                   # R0 helper: keep its body's commands for the path checks
                functions.add(match.group(1))
                line = match.group(2)
            for tokens in split_commands(line):
                in_function = match is not None
                assigns = []
                while tokens and re.match(r"[A-Za-z_]\w*=", tokens[0]):
                    name, value = tokens[0].split("=", 1)
                    assigns.append(name)
                    if value and "$" not in value:
                        variables[name] = value
                    tokens = tokens[1:]
                out.append(Command([expand(t, variables) for t in tokens], line, index, assigns, in_function))
    return out, variables, functions


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

def bash_commands(text: str) -> list[Command]:
    """Commands of the bash blocks only (the inline spans come after them in ``read_runbook``)."""
    n_blocks = len(bash_blocks(text))
    return [cmd for cmd in read_runbook(text)[0] if cmd.block < n_blocks]


def step_sections(text: str) -> list[tuple[str, str]]:
    """(name, text) of every ``## `` section in document order: ``R0``-``R16`` by number, the others by title."""
    out = []
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        title = part.splitlines()[0]
        match = re.match(r"(R\d+)\. ", title)
        out.append((match.group(1) if match else title.strip(), part))
    return out


def section_sources(section: str) -> list[str]:
    """The bash blocks and the inline ``python -I rebuild/...`` spans of one section in document order."""
    found = [(m.start(), m.group(1)) for m in re.finditer(r"^```bash\n(.*?)^```", section, flags=re.S | re.M)]
    outside = re.sub(r"^```.*?^```", lambda m: " " * len(m.group(0)), section, flags=re.S | re.M)
    found += [(m.start(), m.group(1)) for m in re.finditer(r"`([^`\n]+)`", outside) if "python -I rebuild/" in m.group(1)]
    return [source for _, source in sorted(found)]


def chains(line: str) -> list[list[list[str]]]:
    """The ``&&`` chains of one logical line, each a list of commands (``shlex`` tokens). ``;``, ``||``, ``|``,
    ``&`` and redirections end a chain; the parentheses of ``$(...)`` end a command only (a capture stays in its
    chain)."""
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


def chain_commands(chain: list[list[str]], variables: dict[str, str]) -> list[tuple[list[str], list[str]]]:
    """(assigned names, argv with literal variables expanded) of each command of a chain."""
    out = []
    for tokens in chain:
        assigns = []
        while tokens and re.match(r"[A-Za-z_]\w*=", tokens[0]):
            assigns.append(tokens[0].split("=", 1)[0])
            tokens = tokens[1:]
        out.append((assigns, [expand(t, variables) for t in tokens]))
    return out


def step_chains(text: str) -> list[tuple[str, list[tuple[list[str], list[str]]]]]:
    """(step name, commands of one ``&&`` chain) in document order, the R0 helper definitions left out."""
    _, variables, _ = read_runbook(text)
    out = []
    for name, section in step_sections(text):
        for source in section_sources(section):
            for line in logical_lines(source):
                if not HELPER_RE.fullmatch(line):
                    out.extend((name, chain_commands(chain, variables)) for chain in chains(line))
    return out


def folder_command(argv: list[str], main_parser: argparse.ArgumentParser) -> Optional[tuple[str, bool]]:
    """(command, resume) of a ``rebuild/main.py`` call or ``("compare", False)`` of ``compare_runs.py``, else None."""
    _, script, args = python_target(argv)
    if script == "rebuild/tools/compare_runs.py":
        return "compare", False
    if script == "rebuild/main.py":
        ns = parse(main_parser, args, " ".join(argv))
        return ns.command, ns.resume is not None
    return None


def run_output_tests() -> dict[str, set[str]]:
    """test name -> the ``ILORA_RUN_<NAME>`` names it reads: the literal arguments of ``run_dir_env`` and the string
    constants of a ``for`` loop whose variable is passed to it (``tests/vessl/test_run_outputs.py``)."""
    tree = ast.parse(RUN_OUTPUTS.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        loops: dict[str, list[str]] = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.For) and isinstance(sub.target, ast.Name) and isinstance(sub.iter, (ast.Tuple, ast.List)):
                loops[sub.target.id] = [e.value for e in sub.iter.elts
                                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        names: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "run_dir_env" and sub.args:
                arg = sub.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.add(arg.value)
                elif isinstance(arg, ast.Name):
                    names.update(loops.get(arg.id, []))
        out[node.name] = names
    return out


# ----------------------------------------------------------------------------
# §18.8 #4
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M14-4")
def test_commands_are_valid(parsers):
    """Every ``rebuild/main.py`` call parses with main.py's own parser, names one of the six commands and an existing
    profile, and its ``--set`` values load with their types (M0 ``load_config``); ``compare_runs.py`` and
    ``convert_sasrec.py`` calls parse with their parsers; pytest paths exist; ``latest <name>`` is a profile (or
    ``compare``) and the profile of the command before it; shell variables are set before use; the steps R0-R16 and
    the commands come in the order of §18.6.6/§18.7."""
    text = RUNBOOK.read_text(encoding="utf-8")
    assert tuple(re.findall(r"^## (R\d+)\. ", text, flags=re.M)) == STEPS
    cmds, variables, functions = read_runbook(text)
    assert {"latest", "done_ok", "rec", "need", "outputs"} <= functions
    paper, _ = load_config(CONFIG_DIR / "paper_spec.yaml")
    assigned = set(SHELL_VARS)
    seen: list[tuple[str, str]] = []
    made: dict[int, str] = {}
    counts = {"main": 0, "compare": 0, "convert": 0, "pytest": 0, "latest": 0}
    first_pytest = None
    for cmd in cmds:
        argv = cmd.argv
        where = f"RUNBOOK: {cmd.line.strip()}"
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
                assert ns.config is not None, f"{where}: --config missing"
                config = REPO_ROOT / ns.config
                assert config.is_file() and config.parent == CONFIG_DIR, where
                try:
                    cfg, applied = load_config(config, ns.sets)
                    commands.command_requirements(ns.command, cfg)
                except ConfigError as exc:
                    pytest.fail(f"{where}: {exc}")
                assert len(applied) == len(ns.sets), where
                made[cmd.block] = config.stem
                seen.append((ns.command, config.stem))
        elif script == "rebuild/tools/compare_runs.py":
            counts["compare"] += 1
            ns = parse(parsers["compare"], args, where)
            names = [name for name, _ in ns.runs]
            assert len(names) == len(set(names)) and {"E1", "E2", "E3", "E4"} <= set(names), where
            assert {name for name, _ in ns.exports} <= set(names) and ns.cluster and ns.root is None, where
            made[cmd.block] = "compare"
            seen.append(("compare", "compare"))
        elif script == "rebuild/tools/convert_sasrec.py":
            counts["convert"] += 1
            ns = parse(parsers["convert"], args, where)
            assert ns.repo == Path(".") and ns.out == Path(paper.sasrec.state_path).parent, where   # R2 -> profiles
        elif module == "pytest":
            paths = [a for a in args if not a.startswith("-") and a.startswith("rebuild/")]
            assert paths and all((REPO_ROOT / p).exists() for p in paths), where
            if not cmd.in_function:                      # the R3 bundles (outputs() is only defined in R0)
                counts["pytest"] += 1
                first_pytest = len(seen) if first_pytest is None else first_pytest
        elif argv[:1] == ["latest"]:
            counts["latest"] += 1
            name = argv[1]
            assert name in PROFILES or name == "compare", f"{where}: latest {name} is not a profile"
            if cmd.block in made:
                assert name == made[cmd.block], f"{where}: the folder was made by profile {made[cmd.block]}"
        elif argv[:2] == ["conda", "activate"]:
            assert argv[2] in ("ilora-rebuild", "hf-download"), f"{where}: only the new env (D13, 18.5)"
        elif argv[:2] == ["conda", "create"]:
            assert argv[argv.index("-n") + 1] in ("ilora-rebuild", "hf-download") and "python=3.10" in argv, where
        elif argv[:2] == ["git", "clone"]:
            assert not any(a.startswith("--depth") for a in argv), f"{where}: full history needed (R2)"
        elif argv[:1] == ["bash"]:
            assert not any(a.endswith(("setup_vessl.sh", "bootstrap_vessl.sh")) for a in argv), where
        assigned.update(cmd.assigns)
        if argv[:1] == ["rec"]:
            assigned.add(argv[1])
        if argv[:1] == ["export"]:
            assigned.update(a.split("=", 1)[0] for a in argv[1:] if "=" in a)
    assert counts["main"] >= 18 and counts["compare"] == 2 and counts["convert"] == 1 and counts["pytest"] >= 2
    assert counts["latest"] >= 20                 # one after every folder-making command (R4-R15) and the resumes
    positions = [seen.index(item) for item in ORDER]
    assert positions == sorted(positions), [ORDER[i] for i in range(len(ORDER)) if positions[i] != sorted(positions)[i]]
    assert first_pytest is not None and first_pytest <= positions[0]          # R3 tests before R4 check-env
    assert "python -m pip install -r requirements.txt" in text and "pytest==9.2.0" in text
    match = re.search(r"^V=(\w+)\s+#\s*(.+)$", text, flags=re.M)              # R15: the variant names
    assert match, "R15 has no V=<variant> line"
    names = match.group(2).split()
    assert match.group(1) in names and variables["V"] == match.group(1)
    variants = {p[len("variant_"):] for p in PROFILES if p.startswith("variant_")}
    assert set(names) | {"text_only_eval", "user_level_cluster", "bf16"} == variants
    assert "variant_bf16.yaml" in text


@pytest.mark.criterion("M14-4")
def test_output_selectors():
    """Each ``outputs <selector>`` selects tests of ``tests/vessl/test_run_outputs.py`` (pytest ``-k`` substring
    rule) whose ``ILORA_RUN_<NAME>`` folders were recorded with ``rec`` before it, and every test of that file is
    selected by one step, so no run-folder check is left out of the RUNBOOK (§17.6.1, §17.8 #9)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    tests = run_output_tests()
    assert tests, "no tests in test_run_outputs.py"
    recorded: set[str] = set()
    selected: set[str] = set()
    selectors = []
    for cmd in bash_commands(text):
        if cmd.argv[:1] == ["rec"]:
            assert re.fullmatch(r"ILORA_\w+", cmd.argv[1]), cmd.line
            recorded.add(cmd.argv[1])
        elif cmd.argv[:1] == ["outputs"]:
            selector = cmd.argv[1].lower()
            selectors.append(selector)
            assert not [k for k in GENERIC_KEYWORDS if selector in k.lower()], f"outputs {selector} selects everything"
            matched = sorted(name for name in tests if selector in name.lower())
            assert matched, f"outputs {selector} selects no test of {RUN_OUTPUTS.name}"
            for name in matched:
                need = {f"ILORA_RUN_{run_name}" for run_name in tests[name]}
                assert need <= recorded, f"outputs {selector}: {name} reads {sorted(need - recorded)} not recorded yet"
            selected.update(matched)
    assert selectors == ["smoke_gpu", "overfit", "e2_run", "e2_run", "e2_reeval", "e2_export", "cluster", "e4", "e1",
                         "e2_run", "e2_reeval"]                  # the last two: the bf16 block (R7 failure rule)
    assert set(tests) == selected, f"run-folder tests no RUNBOOK step selects: {sorted(set(tests) - selected)}"
    for names in tests.values():
        for name in names:
            assert f"ILORA_RUN_{name}" in text


@pytest.mark.criterion("M14-4")
def test_no_credentials():
    """No token-like value and no assignment to a credential variable (``HF_TOKEN`` is named, never set) [SPEC §0.9]."""
    text = RUNBOOK.read_text(encoding="utf-8")
    assert run.find_secrets([text]) == []
    cmds, _, _ = read_runbook(text)
    for cmd in cmds:
        names = list(cmd.assigns)
        if cmd.argv[:1] == ["export"]:
            names += [a.split("=", 1)[0] for a in cmd.argv[1:]]
        elif cmd.argv[:1] == ["rec"]:
            names.append(cmd.argv[1])
        assert not [n for n in names if SECRET_NAME_RE.search(n)], cmd.line
    assert "HF_TOKEN" in text and not re.search(r"HF_TOKEN\s*=", text)


@pytest.mark.criterion("M14-4")
def test_paths_survive_a_new_shell():
    """§18.6.6 (paths in runbook_paths.sh, restored by ``source`` in R0 after a lost shell): each step read as in a new
    shell after R0. Every ``$NAME`` that a ``rebuild/main.py`` or ``compare_runs.py`` call of R8-R16 or of the bf16
    block reads is set earlier in the same step, by R0, or is an ``ILORA_*`` name that ``rec`` recorded in an earlier
    step; every such ``ILORA_*`` name is checked with ``need`` earlier in the same ``&&`` chain."""
    text = RUNBOOK.read_text(encoding="utf-8")
    restored: set[str] = set()                                  # rec names of earlier steps (runbook_paths.sh)
    r0: set[str] = set(SHELL_VARS)
    assigned: dict[str, set[str]] = {}
    checked = []
    for step, chain in step_chains(text):
        if step not in assigned and step != "R0":
            restored |= {n for s, names in assigned.items() if s != "R0" for n in names if n.startswith("ILORA_")}
        here = assigned.setdefault(step, set())
        needed: set[str] = set()
        for names, argv in chain:
            if argv[:1] == ["need"]:
                needed.update(argv[1:])
            _, script, args = python_target(argv)
            if script in ("rebuild/main.py", "rebuild/tools/compare_runs.py") and (step in LATE_STEPS
                                                                                 or not step.startswith("R")):
                refs = {m for a in args for m in re.findall(r"\$\{?([A-Za-z_]\w*)", a)}
                where = f"{step}: {' '.join(argv)[:120]}"
                assert not refs - here - restored - r0, f"{where}: {sorted(refs - here - restored - r0)} lost in a new shell"
                unchecked = {r for r in refs if r.startswith("ILORA_")} - needed
                assert not unchecked, f"{where}: {sorted(unchecked)} not checked with need first"
                checked.append(step)
            here.update(names)
            if argv[:1] == ["rec"]:
                here.add(argv[1])
            if argv[:1] == ["export"]:
                here.update(a.split("=", 1)[0] for a in argv[1:] if "=" in a)
        if step == "R0":
            r0 |= here
    assert {"R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15"} <= set(checked)
    assert [s for s in checked if not s.startswith("R")], "the bf16 block was not read"


@pytest.mark.criterion("M14-4")
def test_folder_captures_follow_their_command(parsers):
    """``X=$(latest <profile>)`` takes the folder of the command just before it in the same ``&&`` chain, so a command
    that fails before making its folder (``ConfigError``, exit 2) stops the chain instead of recording an older folder;
    or it picks the folder that ``train --resume`` reopens right after it. Every ``done_ok`` in that chain names the
    command of the folder it checks (§18.6.6 ``latest``, ``done_ok``, ``rec``)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    seen = 0
    for step, chain in step_chains(text):
        commands_ = [(names, argv) for names, argv in chain if argv]
        for i, (_, argv) in enumerate(commands_):
            if argv[:1] == ["done_ok"]:
                assert len(argv) == 3 and argv[2] in (*commands.COMMANDS, "compare"), f"{step}: {argv}"
            if argv[:1] != ["latest"]:
                continue
            before = folder_command(commands_[i - 1][1], parsers["main"]) if i else None
            after = folder_command(commands_[i + 1][1], parsers["main"]) if i + 1 < len(commands_) else None
            if before is not None and not before[1]:
                maker = before[0]
            else:
                assert after == ("train", True), f"{step}: latest {argv[1]} does not follow the command of its folder"
                maker = "train"
            checks = [a for _, a in commands_[i + 1:] if a[:1] == ["done_ok"]]
            assert checks and checks[0][2] == maker, f"{step}: latest {argv[1]} is not checked as a {maker} folder"
            seen += 1
    assert seen >= 25


@pytest.mark.criterion("M14-4")
def test_resume_only_after_the_process_ended():
    """R7: a lost shell does not stop a run inside tmux, so it is no reason to resume; the step reattaches with
    ``tmux attach -t rebuild`` and, in the block that resumes, first checks that no ``rebuild/main.py`` process and no
    GPU memory remain; the run.lock refusal of ``train --resume`` (M0) and the ``interrupted`` segment (M10) are named.
    R12 and R13 resume under the same condition."""
    sections = dict(step_sections(RUNBOOK.read_text(encoding="utf-8")))
    r7 = sections["R7"]
    assert "셸 끊김" not in r7
    (block,) = [b for b in bash_blocks(r7) if "--resume" in b]
    before = block[:block.index("--resume")]
    assert "pgrep -af 'rebuild/main.py'" in before and "nvidia-smi --query-gpu=memory.used" in before
    assert r7.index("tmux attach -t rebuild") < r7.index(block)
    assert "run.lock" in r7 and "is in use by a live process" in r7 and "interrupted" in r7
    for step in ("R12", "R13"):
        assert "R7의 조건" in sections[step] and "train --resume" in sections[step]


@pytest.mark.criterion("M14-4")
def test_r1_download_sets_llm_path():
    """On a fresh workspace R1 comes before R0: the block that downloads LLaMA exports LLM_PATH itself, with R0's
    value, before the line that reads it."""
    sections = dict(step_sections(RUNBOOK.read_text(encoding="utf-8")))
    value = re.search(r"^export LLM_PATH=(\S+)", sections["R0"], flags=re.M).group(1)
    (block,) = [b for b in bash_blocks(sections["R1"]) if 'os.environ["LLM_PATH"]' in b]
    lines = block.splitlines()
    exported = [i for i, line in enumerate(lines) if re.match(rf"export LLM_PATH={re.escape(value)}(\s|$)", line)]
    used = [i for i, line in enumerate(lines) if 'os.environ["LLM_PATH"]' in line]
    assert exported and used and exported[0] < used[0]


@pytest.mark.criterion("M14-4")
def test_r3_skip_rule_matches_the_bundle():
    """The R3 CPU bundle (``pytest rebuild/tests`` without ILORA_VESSL) also collects tests/vessl, whose tests all skip
    with VESSL_REQUIRED (tests/vessl/conftest.py); R3 allows exactly that skip in it and no other (§17.6.1, §17.8 #2)."""
    r3 = dict(step_sections(RUNBOOK.read_text(encoding="utf-8")))["R3"]
    cmds, _, _ = read_runbook(r3)
    (cpu,) = [c for c in cmds if c.argv[:4] == ["python", "-I", "-m", "pytest"] and "ILORA_VESSL" not in c.assigns]
    assert "rebuild/tests" in cpu.argv and not [a for a in cpu.argv if a.startswith("--ignore")]
    assert "CPU 묶음의 skip은 0" not in r3 and "CPU 묶음에서 `VESSL_REQUIRED` 밖의 skip은 0" in r3
    assert "VESSL_REQUIRED" in (REBUILD_DIR / "tests" / "vessl" / "conftest.py").read_text(encoding="utf-8")


@pytest.mark.criterion("M14-4")
def test_bf16_branch_commands(parsers):
    """The bf16 fallback of R7 (D9, §18.6.4) has its own commands: E2 with variant_bf16.yaml (``latest variant_bf16``),
    the R8 re-evaluations with variant_bf16.yaml, E3, E4 and E1 with ``model.torch_dtype=bfloat16`` and
    ``train.precision=bf16_amp``: every training and evaluation it starts loads as bf16; the R15 variants are told to
    add the same two keys."""
    sections = dict(step_sections(RUNBOOK.read_text(encoding="utf-8")))
    (title,) = [name for name in sections if name.startswith("bf16")]
    runs = []
    for cmd in read_runbook(sections[title])[0]:
        _, script, args = python_target(cmd.argv)
        if script != "rebuild/main.py":
            continue
        ns = parse(parsers["main"], args, cmd.line)
        if ns.resume is not None:
            continue
        cfg, _ = load_config(REPO_ROOT / ns.config, ns.sets)
        assert (cfg.model.torch_dtype, cfg.train.precision) == ("bfloat16", "bf16_amp"), cmd.line
        runs.append((ns.command, ns.config.stem))
    assert runs == [("train", "variant_bf16"), ("eval", "variant_bf16"), ("eval", "variant_bf16"),
                    ("eval", "variant_text_only_eval"), ("train", "cluster_hard"), ("train", "llara_baseline")]
    assert "latest variant_bf16" in sections[title]
    assert " ".join(f"--set {s}" for s in BF16_SETS) in sections[title].split("R15")[-1]


@pytest.mark.criterion("M14-4")
def test_r15_example_variant():
    """The R15 example variant is not released_lr, which the fork saw diverge in fp16 at lr 8e-4 (fork
    IMPLEMENTATION_NOTES §11i); R15 says how a released_lr run is expected to stop and how its failed folder is
    reported (NOT_RUN with the TrainingError reason)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    _, variables, _ = read_runbook(text)
    assert variables["V"] != "released_lr"
    r15 = dict(step_sections(text))["R15"]
    assert "§11i" in r15 and "TrainingError" in r15 and "variant_released_lr" in r15 and "NOT_RUN" in r15

# ----------------------------------------------------------------------------
# §18.8 #5-#6: computed training lengths and the overfit rows
# ----------------------------------------------------------------------------
# §18.6.5 table (seed 1234; M1 5.6.7 smoke selection, M10 14.6.2 lengths, M4 8.6.5 curriculum) and the VESSL M10-16
# and M10-19 runs of §17.6.12: profile and --set, train samples, NB, tail, steps per epoch, T, w, micro-batches, hybrid
# micro-batches per epoch (None: no value in the design), E4 cluster lookups = micro-batches x micro-batch size
# (§17.6.6 S5, §17.6.7, RUNBOOK R12; None: no value in the design).
DERIVED = {
    "smoke_cpu": (("smoke_cpu",), 156, 39, 3, 10, 20, 5, 78, (10, 28), None),
    "smoke_cpu_1epoch": (("smoke_cpu", "train.epochs=1"), 156, 39, 3, 10, 10, 2, 39, (23,), 156),
    "smoke_gpu_100": (("smoke_gpu_100",), 7_230, 903, 7, 57, 114, 5, 1_806, (218, 670), None),
    "smoke_gpu_100_1epoch": (("smoke_gpu_100", "train.epochs=1"), 7_230, 903, 7, 57, 57, 2, 903, (428,), 7_224),
    "overfit_gpu": (("overfit_gpu",), 16, 4, 0, 4, 100, 5, 100, (4,) * 25, None),
    "paper_spec": (("paper_spec",), 66_126, 8_265, 9, 517, 2_585, 129, 41_325, (846, 2_550, 4_076, 5_766, 7_436),
                   330_600),
    "vessl_nan_step": (("smoke_gpu_100", "data.smoke_users=[1,10,10]", "train.epochs=1", "train.grad_accum=2",
                        "train.warmup_divisor=4"), 132, 16, 0, 8, 8, 2, 16, None, None),
    "vessl_resume": (("smoke_gpu_100", "data.smoke_users=[1,10,10]", "train.epochs=2", "train.grad_accum=2",
                      "train.warmup_divisor=4", "eval.batch_invariance_check=false"), 132, 16, 0, 8, 16, 4, 32, None,
                     None),
}


@pytest.mark.criterion("M14-5")
@pytest.mark.parametrize("case", list(DERIVED))
def test_derived_lengths(case, full_data, profile_cfg):
    """The §18.6.5 numbers recomputed with M1 ``select_smoke``, M10 ``step_counts``/``window_sizes`` and M4
    ``build_batch_plan``: samples, NB, tail, steps per epoch, T, w, micro-batches, hybrid micro-batches per epoch;
    the run-record counts the RUNBOOK checks (R5, R7, R12) are the same numbers."""
    from ilora.data import select_smoke
    from ilora.prompts import build_batch_plan
    from ilora.train import step_counts

    (profile, *sets), n, nb, tail, per_epoch, total, warmup, micro, hybrid, lookups = DERIVED[case]
    cfg = profile_cfg(profile, *sets)
    data = full_data if cfg.data.subset == "full" else select_smoke(
        full_data, cfg.run.seed, tuple(cfg.data.smoke_users), train_rows=tuple(cfg.data.train_rows))
    sc = step_counts(len(data.train), cfg.train)
    assert (sc.n_samples, sc.micro_per_epoch, sc.tail, sc.steps_per_epoch, sc.total_steps, sc.warmup_steps,
            sc.total_micro) == (n, nb, tail, per_epoch, total, warmup, micro)
    windows = sc.window_sizes()
    assert len(windows) == per_epoch and sum(windows) == nb
    assert windows == (cfg.train.grad_accum,) * (nb // cfg.train.grad_accum) + ((tail,) if tail else ())   # D24
    plans = [build_batch_plan(data.train, epoch, cfg) for epoch in range(cfg.train.epochs)]
    assert [plan.batches_per_epoch for plan in plans] == [nb] * cfg.train.epochs
    assert all(plan.total == micro and plan.entries[0].tau == e * nb + 1 for e, plan in enumerate(plans))
    if hybrid is not None:
        assert tuple(plan.hybrid_count() for plan in plans) == hybrid
    if lookups is not None:
        assert micro * cfg.train.micro_batch_size == lookups
    runbook = RUNBOOK.read_text(encoding="utf-8")
    if case == "paper_spec":                                   # R7, R12, R13
        assert all(f"{value:,}" in runbook for value in (micro, total, lookups))
    if case == "smoke_gpu_100":                                # R5
        assert all(f"{value:,}" in runbook for value in (micro, total, sum(hybrid), DERIVED["smoke_gpu_100_1epoch"][-1]))


@pytest.mark.criterion("M14-6")
def test_overfit_rows(full_data, profile_cfg):
    """``overfit_gpu`` ``data.train_rows`` are the last train window of each user 0-15 (len_seq 10), the 16 targets
    are distinct and none is item 266 (empty code title); the profile's smoke selection gives exactly them
    (§17.6.8, §18.6.5)."""
    from ilora.data import select_smoke

    cfg = profile_cfg("overfit_gpu")
    last = {}
    for sample in full_data.train:                             # row order: the later window of a user wins
        if sample.user_id < 16:
            last[sample.user_id] = sample
    chosen = [last[user] for user in range(16)]
    assert cfg.data.train_rows == tuple(s.row for s in chosen) and len(cfg.data.train_rows) == 16
    assert all(len(s.history) == 10 for s in chosen)
    targets = [s.target for s in chosen]
    assert len(set(targets)) == 16 and 266 not in targets
    assert cfg.data.smoke_users[0] == 16 and cfg.data.subset == "smoke"
    data = select_smoke(full_data, cfg.run.seed, tuple(cfg.data.smoke_users), train_rows=cfg.data.train_rows)
    assert [s.sample_id for s in data.train] == [s.sample_id for s in chosen]
    assert sorted({s.user_id for s in data.train}) == list(range(16))

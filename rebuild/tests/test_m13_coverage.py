"""M13 시험 기반 점검 (DESIGN.md §17.6.3, §17.8 #1, #2, #4, #5; [SPEC §21]).

Static checks over ``rebuild/``: every completion criterion of DESIGN.md (M0-M15, M13 and M14 included) has a
test with ``pytest.mark.criterion`` and no test names a criterion the design lacks; every skip reason in the test
code is a documented constant of the two conftest modules; ``repo_guard`` is a session autouse fixture; no file
under ``rebuild/`` imports repository code (only ``ilora``, the standard library, installed packages and the
test-local modules ``refs_upstream`` and ``conftest``, §17.6.4); ``rebuild/`` compiles; every function of
``tests/refs_upstream.py`` names its ``U``/``F`` source lines. The checks are static (``ast``), so ``-k``/``-m``
selections do not change their result.
"""
from __future__ import annotations

import ast
import compileall
import importlib.metadata
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
TESTS_DIR = REBUILD_DIR / "tests"
DESIGN = REBUILD_DIR / "DESIGN.md"
CONFTESTS = (TESTS_DIR / "conftest.py", TESTS_DIR / "vessl" / "conftest.py")

CHAPTER_RE = re.compile(r"^## (\d+)\. M(\d+) ")              # "## 17. M13 테스트"
ITEM_RE = re.compile(r"^(\d+)\. ")                            # "1. (CPU) ..." under "### c.8 완료 기준"
CRITERION_ID_RE = re.compile(r"M(0|[1-9][0-9]*)-([1-9][0-9]*)")
TOTAL_RE = re.compile(r"^- 합계 (\d+)개 \(([^)]*)\)")            # 17.6.9: "- 합계 187개 (M0 6, M1 14, ...)"
# §17.6.3: repository folder names that must never be imported (1.3)
FORBIDDEN_IMPORTS = ("model", "data", "recommender", "optims", "SASRecModules_ori", "debug", "analysis", "scripts",
                     "measurement", "main")
# Test-local modules (§17.6.4 refs_upstream; pytest's conftest modules), importable only from tests/.
TEST_LOCAL_MODULES = ("refs_upstream", "conftest")
SKIP_CALLS = (("pytest", "skip"), ("pytest", "importorskip"), ("mark", "skip"), ("mark", "skipif"))
SOURCE_RE = re.compile(r"``(U|F|L|U@0a69c15) [A-Za-z0-9_./-]+\.py:\d+")   # §0.2 citation form


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def design_criteria(text: str) -> dict[int, list[int]]:
    """Module number -> item numbers of its "### c.8 완료 기준" section (lines ``^\\d+\\. ``)."""
    out: dict[int, list[int]] = defaultdict(list)
    chapter, module, inside = None, None, False
    for line in text.splitlines():
        match = CHAPTER_RE.match(line)
        if match:
            chapter, module, inside = match.group(1), int(match.group(2)), False
            continue
        if line.startswith("## "):
            chapter, module, inside = None, None, False
            continue
        if chapter is not None and line.startswith(f"### {chapter}.8 완료 기준"):
            inside = True
            continue
        if line.startswith("### ") or line.startswith("#### "):
            inside = False
            continue
        item = ITEM_RE.match(line)
        if inside and item:
            out[module].append(int(item.group(1)))
    return dict(out)


def _criterion_call(node: ast.AST) -> bool:
    """``pytest.mark.criterion(...)`` (or any ``<x>.mark.criterion(...)``)."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "criterion"
            and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "mark")


def collect_markers() -> tuple[dict[str, list[str]], list[str]]:
    """criterion id -> ["file:line", ...] over rebuild/tests/**/*.py, and the problems found (non-literal ids)."""
    found: dict[str, list[str]] = defaultdict(list)
    problems = []
    for path in _python_files(TESTS_DIR):
        for node in ast.walk(_parse(path)):
            if not _criterion_call(node):
                continue
            where = f"{path.relative_to(REBUILD_DIR).as_posix()}:{node.lineno}"
            if node.keywords or not node.args:
                problems.append(f"{where}: criterion() needs positional string ids only")
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and CRITERION_ID_RE.fullmatch(arg.value):
                    found[arg.value].append(where)
                else:
                    problems.append(f"{where}: criterion id {ast.unparse(arg)} is not a literal 'M<n>-<k>'")
    return dict(found), problems




@pytest.mark.criterion("M13-1")
def test_every_criterion_has_a_test():
    """17.8 #1: the criterion ids of DESIGN.md (every "### c.8 완료 기준" item of chapters M0-M15) equal the ids of
    the criterion markers in rebuild/tests: no criterion without a test, no test with an unknown id. The 17.6.9
    total (187 for M0-M12 and M15) is checked against the same parse."""
    text = DESIGN.read_text(encoding="utf-8")
    items = design_criteria(text)
    assert sorted(items) == list(range(16)), f"modules with a completion-criteria section: {sorted(items)}"
    for module, numbers in items.items():
        assert numbers == list(range(1, len(numbers) + 1)), f"M{module} criteria are not numbered 1..n: {numbers}"
    totals = [TOTAL_RE.match(line) for line in text.splitlines()]
    (total,) = [m for m in totals if m]
    per_module = {int(m): int(n) for m, n in re.findall(r"M(\d+) (\d+)", total.group(2))}
    assert per_module == {m: len(items[m]) for m in per_module}, "17.6.9 per-module counts differ from the sections"
    assert int(total.group(1)) == sum(per_module.values()) and 13 not in per_module and 14 not in per_module
    documented = {f"M{module}-{n}" for module, numbers in items.items() for n in numbers}
    assert len(documented) == int(total.group(1)) + len(items[13]) + len(items[14])     # 187 + M13 + M14
    found, problems = collect_markers()
    assert not problems, "\n".join(problems)
    missing = sorted(documented - set(found), key=_criterion_key)
    extra = sorted(set(found) - documented, key=_criterion_key)
    assert not missing and not extra, (f"criteria without a test: {missing}\n"
                                       f"tests naming unknown criteria: {[(i, found[i]) for i in extra]}")


def _criterion_key(cid: str) -> tuple[int, int]:
    module, number = CRITERION_ID_RE.fullmatch(cid).groups()
    return int(module), int(number)


# ----------------------------------------------------------------------------
# Skip reasons (17.8 #2)
# ----------------------------------------------------------------------------
def documented_reasons() -> dict[str, str]:
    """Name -> text of every constant listed in ``DOCUMENTED_SKIP_REASONS`` of the two conftest modules."""
    out: dict[str, str] = {}
    for path in CONFTESTS:
        strings: dict[str, str] = {}
        listed: list[str] = []
        for node in _parse(path).body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                continue
            name = node.targets[0].id
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                strings[name] = node.value.value
            if name == "DOCUMENTED_SKIP_REASONS":
                assert isinstance(node.value, ast.Tuple), path
                listed = [e.id for e in node.value.elts if isinstance(e, ast.Name)]
                assert len(listed) == len(node.value.elts), f"{path}: DOCUMENTED_SKIP_REASONS must list names"
        assert listed, f"{path}: no DOCUMENTED_SKIP_REASONS"
        for name in listed:
            assert name in strings, f"{path}: {name} is not a string constant of the module"
            out[name] = strings[name]
    return out


def _skip_call(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    owner = node.func.value
    owner_name = owner.id if isinstance(owner, ast.Name) else owner.attr if isinstance(owner, ast.Attribute) else None
    return (owner_name, node.func.attr) in SKIP_CALLS


@pytest.mark.criterion("M13-2")
def test_skip_reasons_documented():
    """17.8 #2 (static part; "failures 0" is the run itself): every ``pytest.skip``, ``pytest.mark.skip``,
    ``pytest.mark.skipif`` and ``pytest.importorskip`` in rebuild/tests gives its reason as a documented constant of
    tests/conftest.py or tests/vessl/conftest.py (SKIP_NO_TOKENIZER, SKIP_NO_SASREC, SKIP_NO_GIT_HISTORY,
    VESSL_REQUIRED, SKIP_NO_RUN, SKIP_NO_LLM, SKIP_NO_PL) or its exact text."""
    reasons = documented_reasons()
    assert {"SKIP_NO_TOKENIZER", "SKIP_NO_SASREC", "SKIP_NO_GIT_HISTORY", "VESSL_REQUIRED", "SKIP_NO_RUN",
            "SKIP_NO_LLM", "SKIP_NO_PL"} <= set(reasons)
    texts = set(reasons.values())
    problems, seen = [], 0
    for path in _python_files(TESTS_DIR):
        for node in ast.walk(_parse(path)):
            if not _skip_call(node):
                continue
            seen += 1
            where = f"{path.relative_to(REBUILD_DIR).as_posix()}:{node.lineno}"
            keyword = {k.arg: k.value for k in node.keywords}
            name = node.func.attr
            if name == "importorskip":
                reason = keyword.get("reason")
            elif name == "skipif":
                reason = keyword.get("reason")
            else:
                reason = keyword.get("reason", node.args[0] if node.args else None)
            ok = (isinstance(reason, ast.Name) and reason.id in reasons) or \
                 (isinstance(reason, ast.Constant) and reason.value in texts)
            if not ok:
                problems.append(f"{where}: {ast.unparse(node)[:120]}")
    assert seen > 0, "no skip call found: the scan is broken"
    assert not problems, "skip reasons that are not documented constants:\n" + "\n".join(problems)


@pytest.mark.criterion("M13-4")
def test_repo_guard_autouse(request):
    """17.8 #4: ``repo_guard`` of tests/conftest.py is a session-scoped autouse fixture (so every session compares
    ``git status --porcelain`` and the runs/rebuild/ listing, assets/ and vessl_tests_* excluded, at teardown)."""
    assert "repo_guard" in request.fixturenames
    tree = _parse(TESTS_DIR / "conftest.py")
    (guard,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "repo_guard"]
    decorators = [d for d in guard.decorator_list if isinstance(d, ast.Call) and ast.unparse(d.func) == "pytest.fixture"]
    assert len(decorators) == 1
    options = {k.arg: ast.literal_eval(k.value) for k in decorators[0].keywords}
    assert options == {"scope": "session", "autouse": True}
    (state,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_repo_state"]
    source = ast.unparse(state)
    assert "'--porcelain'" in source and "'assets'" in source and "'vessl_tests_'" in source
    calls = {ast.unparse(n.func) for n in ast.walk(guard) if isinstance(n, ast.Call)}
    assert {"_repo_state", "pytest.fail"} <= calls


# ----------------------------------------------------------------------------
# Imports, compilation and the reference copies (17.8 #5, [SPEC §21])
# ----------------------------------------------------------------------------
def repository_names() -> set[str]:
    """Top-level importable names of the repository outside rebuild/: the §17.6.3 list plus every root ``*.py`` file
    and every root folder that holds Python files (``model``, ``data``, ``recommender``, ``scripts``, ...)."""
    names = set(FORBIDDEN_IMPORTS)
    for path in REPO_ROOT.iterdir():
        if path.name in ("rebuild",) or path.name.startswith((".", "__")):
            continue
        if path.is_file() and path.suffix == ".py":
            names.add(path.stem)
        elif path.is_dir() and any(path.glob("*.py")):
            names.add(path.name)
    return names


def requirement_names() -> set[str]:
    """Import-style names of the repository's pinned requirements (``requirements.txt``, the VESSL env of M14 R1):
    ``pytorch_lightning`` is installed there for M10 14.8 #18 but not in the local test env."""
    names = set()
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*==", line)
        if match:
            names.add(match.group(1).lower().replace("-", "_").replace(".", "_"))
    return names


def installed_package(name: str) -> bool:
    """``name`` is importable from outside the repository (``packages_distributions`` of Python 3.10 misses wheels
    without top_level.txt such as scipy and threadpoolctl, so ``find_spec`` decides as well)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    places = [spec.origin] if spec.origin else list(spec.submodule_search_locations or [])
    return bool(places) and not any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in places if p)


def _imports(tree: ast.Module) -> list[tuple[int, str, int]]:
    """(line, top-level name, relative level) of every import statement, including those inside functions."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, alias.name.split(".")[0], 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append((node.lineno, (node.module or "").split(".")[0], node.level))
    return out


@pytest.mark.criterion("M13-5")
def test_no_repo_imports():
    """17.6.3 / 17.8 #5: every ``.py`` under rebuild/ (tests included) imports only ``ilora``, the standard library and
    installed packages (tests/ also the test-local ``refs_upstream`` and ``conftest``, §17.6.4); repository names such
    as ``model``, ``data``, ``recommender``, ``optims``, ``main`` never appear. Relative imports only inside the
    ``ilora`` package. (``tools/convert_sasrec.py`` loads the original git blob with ``importlib.import_module`` from
    its temporary folder, M3 7.6.1 step 5: a call, not an import statement.)"""
    forbidden = repository_names()
    assert set(FORBIDDEN_IMPORTS) <= forbidden
    allowed = {"ilora"} | set(sys.stdlib_module_names) | set(importlib.metadata.packages_distributions()) | \
        requirement_names()
    problems, files = [], _python_files(REBUILD_DIR)
    assert REBUILD_DIR / "ilora" / "smoke.py" in files and REBUILD_DIR / "main.py" in files
    for path in files:
        rel = path.relative_to(REBUILD_DIR)
        in_package, in_tests = rel.parts[0] == "ilora", rel.parts[0] == "tests"
        for line, name, level in _imports(_parse(path)):
            where = f"{rel.as_posix()}:{line}"
            if level:
                if not in_package:
                    problems.append(f"{where}: relative import outside the ilora package")
                continue
            if name in forbidden:
                problems.append(f"{where}: imports the repository module {name!r}")
            elif name in TEST_LOCAL_MODULES:
                if not in_tests:
                    problems.append(f"{where}: {name!r} is test-local (tests/ only)")
            elif name not in allowed and not installed_package(name):
                problems.append(f"{where}: {name!r} is neither ilora, the standard library nor an installed package")
    assert not problems, "\n".join(problems)


@pytest.mark.criterion("M13-5")
def test_compileall():
    """[SPEC §21] / 17.8 #5: ``compileall.compile_dir(rebuild/, quiet=1)`` succeeds."""
    assert compileall.compile_dir(str(REBUILD_DIR), quiet=1)


@pytest.mark.criterion("M13-5")
def test_refs_upstream_sources():
    """17.6.4 / 17.8 #5: every function of tests/refs_upstream.py has a docstring that names its source lines in the
    0.2 form (``U <path>:<line>`` or ``F <path>:<line>``) and says it is for comparison only (대조 전용); the module
    imports no repository code and nothing under ilora/ imports it."""
    tree = _parse(TESTS_DIR / "refs_upstream.py")
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert {"first_line_and_hr1", "code_prompt", "code_pair_input_and_labels", "wrap_emb_loop", "upstream_lr",
            "moelora_einsum", "fork_sample_idx"} <= {f.name for f in functions}          # the seven of 17.6.4
    problems = []
    for function in functions:
        doc = ast.get_docstring(function) or ""
        if not SOURCE_RE.search(doc):
            problems.append(f"{function.name}: no U/F source citation")
        if "대조 전용" not in doc:
            problems.append(f"{function.name}: does not say it is for comparison only")
    assert not problems, problems
    for path in _python_files(REBUILD_DIR / "ilora"):
        assert all(name != "refs_upstream" for _, name, _ in _imports(_parse(path))), path

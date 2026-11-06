"""iLoRA 재구현 실행 진입점 (DESIGN.md §4.6.1, §17.6.12 제안 3).

    python rebuild/main.py <command> --config <profile.yaml> [--set section.key=value ...]
    python rebuild/main.py train --resume <run folder>

Commands: train | eval | export | cluster | smoke | check-env. This file only
parses arguments, loads the profile and calls ``ilora.commands.run_command``;
it puts ``rebuild/`` on ``sys.path`` itself so that it works from any working
directory and under ``python -I``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REBUILD_DIR = Path(__file__).resolve().parent
if str(REBUILD_DIR) not in sys.path:
    sys.path.insert(0, str(REBUILD_DIR))

from ilora import commands  # noqa: E402  (needs the sys.path entry above)
from ilora.config import ConfigError, load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="iLoRA re-implementation (rebuild/DESIGN.md). Commands: " + " | ".join(commands.COMMANDS)
        + ". Run folders are created under runs/rebuild/<profile>_<UTC>/.",
    )
    parser.add_argument("command", choices=commands.COMMANDS,
                        help="train, eval, export, cluster, smoke or check-env")
    parser.add_argument("--config", type=Path, metavar="PROFILE_YAML",
                        help="profile file, e.g. rebuild/configs/paper_spec.yaml")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override one key with a YAML value; repeatable")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR",
                        help="train only: reopen a failed or unfinished run folder (no --config/--set; "
                             "the folder's config.yaml is used)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    full_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    try:
        if args.resume is not None:
            if args.command != "train":
                parser.error("--resume is only for train")
            if args.config is not None or args.sets:
                parser.error("--resume takes no --config or --set (the run folder's config.yaml is used)")
            run_dir = args.resume.resolve()
            cfg, profile = commands.load_run_config(run_dir)
            commands.run_command("train", cfg, [], profile=profile, root=run_dir.parent, argv=full_argv,
                                 resume=run_dir)
            return 0
        if args.config is None:
            parser.error("--config is required")
        cfg, _ = load_config(args.config, args.sets)
        commands.run_command(args.command, cfg, args.sets, profile=args.config.stem, argv=full_argv,
                             profile_path=args.config.resolve())
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

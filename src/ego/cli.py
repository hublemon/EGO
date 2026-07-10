"""Command-line scaffold for the EGO research project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _run_placeholder(args: argparse.Namespace) -> int:
    print(f"EGO command: {args.stage} {args.command}")
    print(f"Config: {args.config}")
    print("Status: not implemented. This scaffold only parses arguments.")
    return 0


def _add_config_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    stage: str,
    help_text: str,
) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML configuration file.",
    )
    parser.set_defaults(func=_run_placeholder, stage=stage, command=name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ego",
        description="EGO research project command-line scaffold.",
    )
    stage_parsers = parser.add_subparsers(dest="stage", metavar="stage")

    step1 = stage_parsers.add_parser(
        "step1",
        help="V-JEPA2 action anticipation commands.",
    )
    step1_commands = step1.add_subparsers(dest="command", metavar="command")
    _add_config_command(step1_commands, "prepare", "step1", "Prepare Step 1 data.")
    _add_config_command(step1_commands, "train", "step1", "Train Step 1 models.")
    _add_config_command(step1_commands, "infer", "step1", "Run Step 1 inference.")
    _add_config_command(step1_commands, "evaluate", "step1", "Evaluate Step 1.")

    step2 = stage_parsers.add_parser(
        "step2",
        help="VLM alignment commands.",
    )
    step2_commands = step2.add_subparsers(dest="command", metavar="command")
    _add_config_command(step2_commands, "build-data", "step2", "Build Step 2 data.")
    _add_config_command(step2_commands, "sft", "step2", "Run SFT scaffold.")
    _add_config_command(step2_commands, "grpo-noun", "step2", "Run noun-stage GRPO scaffold.")
    _add_config_command(step2_commands, "grpo-action", "step2", "Run action-stage GRPO scaffold.")
    _add_config_command(step2_commands, "infer", "step2", "Run Step 2 inference.")
    _add_config_command(step2_commands, "evaluate", "step2", "Evaluate Step 2.")

    step3 = stage_parsers.add_parser(
        "step3",
        help="Memory-context dynamic planning commands.",
    )
    step3_commands = step3.add_subparsers(dest="command", metavar="command")
    _add_config_command(step3_commands, "run", "step3", "Run planning scaffold.")
    _add_config_command(step3_commands, "evaluate", "step3", "Evaluate planning scaffold.")

    pipeline = stage_parsers.add_parser(
        "pipeline",
        help="End-to-end pipeline commands.",
    )
    pipeline_commands = pipeline.add_subparsers(dest="command", metavar="command")
    _add_config_command(pipeline_commands, "run", "pipeline", "Run pipeline scaffold.")
    _add_config_command(pipeline_commands, "smoke-test", "pipeline", "Run pipeline smoke-test scaffold.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""VGB Framework CLI — unified entry point for all operations.

Provides subcommands: train, eval, infer, export, resume,
inspect-checkpoint, validate-config.

Each subcommand parses args → loads config → bootstraps runtime → executes.

Validates: Requirements 11.1, 11.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by multiple subcommands."""
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON configuration file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Base configuration profile (e.g. vsn_small, vsn_base, vsn_large)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="CLI overrides in dotted.key=value format",
    )


def _build_config(args: argparse.Namespace):
    """Load and validate configuration from CLI args.

    Returns:
        Validated FullConfig instance.
    """
    from vgb.config.loader import build_full_config

    return build_full_config(
        config_path=Path(args.config),
        profile=getattr(args, "profile", None),
        overrides=getattr(args, "overrides", None) or [],
    )


def _bootstrap(config):
    """Bootstrap the runtime (seeds, logging, device, distributed).

    Returns:
        RuntimeContext instance.
    """
    from vgb.runtime.bootstrap import bootstrap

    return bootstrap(config.runtime)


# ── Subcommand handlers ──────────────────────────────────────────────


def _handle_train(args: argparse.Namespace) -> None:
    """Handle the 'train' subcommand."""
    config = _build_config(args)
    ctx = _bootstrap(config)
    print(
        f"[vgb train] Starting training on {ctx.device} "
        f"(world_size={ctx.world_size}) with config: {args.config}"
    )
    # Actual training is delegated to vgb.training.trainer
    from vgb.training.trainer import Trainer

    trainer = Trainer(config=config, context=ctx)
    trainer.fit()


def _handle_eval(args: argparse.Namespace) -> None:
    """Handle the 'eval' subcommand."""
    config = _build_config(args)
    ctx = _bootstrap(config)
    checkpoint = getattr(args, "checkpoint", None)
    print(
        f"[vgb eval] Evaluating checkpoint={checkpoint} "
        f"on {ctx.device} with config: {args.config}"
    )


def _handle_infer(args: argparse.Namespace) -> None:
    """Handle the 'infer' subcommand."""
    config = _build_config(args)
    ctx = _bootstrap(config)
    source = getattr(args, "source", None)
    print(
        f"[vgb infer] Running inference from source={source} "
        f"on {ctx.device} with config: {args.config}"
    )


def _handle_export(args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand."""
    config = _build_config(args)
    ctx = _bootstrap(config)
    checkpoint = getattr(args, "checkpoint", None)
    output_dir = getattr(args, "output_dir", None)
    print(
        f"[vgb export] Exporting checkpoint={checkpoint} "
        f"to output_dir={output_dir} with config: {args.config}"
    )


def _handle_resume(args: argparse.Namespace) -> None:
    """Handle the 'resume' subcommand."""
    config = _build_config(args)
    ctx = _bootstrap(config)
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    print(
        f"[vgb resume] Resuming training from {checkpoint_dir} "
        f"on {ctx.device} with config: {args.config}"
    )


def _handle_inspect_checkpoint(args: argparse.Namespace) -> None:
    """Handle the 'inspect-checkpoint' subcommand."""
    checkpoint_path = args.path
    print(f"[vgb inspect-checkpoint] Inspecting checkpoint: {checkpoint_path}")
    # Inspection does not require full bootstrap
    from vgb.runtime.checkpointing import inspect_checkpoint

    info = inspect_checkpoint(checkpoint_path)
    for key, value in info.items():
        print(f"  {key}: {value}")


def _handle_validate_config(args: argparse.Namespace) -> None:
    """Handle the 'validate-config' subcommand."""
    from vgb.config.loader import build_full_config, ConfigLoadError
    from vgb.config.schema import VGBConfigError

    try:
        config = build_full_config(
            config_path=Path(args.config),
            profile=getattr(args, "profile", None),
            overrides=getattr(args, "overrides", None) or [],
        )
        print(f"[vgb validate-config] Configuration is valid: {args.config}")
    except (ConfigLoadError, VGBConfigError) as e:
        print(f"[vgb validate-config] INVALID: {e}", file=sys.stderr)
        sys.exit(1)


# ── Parser construction ──────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="vgb",
        description="VGB Framework CLI — train, evaluate, infer, and export VSN models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ──
    train_parser = subparsers.add_parser("train", help="Train a VSN model")
    _add_common_args(train_parser)

    # ── eval ──
    eval_parser = subparsers.add_parser("eval", help="Evaluate a VSN model")
    _add_common_args(eval_parser)
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to evaluate",
    )

    # ── infer ──
    infer_parser = subparsers.add_parser("infer", help="Run inference with a VSN model")
    _add_common_args(infer_parser)
    infer_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to bundle or checkpoint for inference",
    )

    # ── export ──
    export_parser = subparsers.add_parser(
        "export", help="Export model to inference bundle"
    )
    _add_common_args(export_parser)
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to export",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Output directory for the exported bundle",
    )

    # ── resume ──
    resume_parser = subparsers.add_parser(
        "resume", help="Resume training from checkpoint"
    )
    _add_common_args(resume_parser)
    resume_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        dest="checkpoint_dir",
        help="Directory containing the checkpoint to resume from",
    )

    # ── inspect-checkpoint ──
    inspect_parser = subparsers.add_parser(
        "inspect-checkpoint", help="Inspect a saved checkpoint"
    )
    inspect_parser.add_argument(
        "path",
        type=str,
        help="Path to the checkpoint file or directory",
    )

    # ── validate-config ──
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a configuration file"
    )
    _add_common_args(validate_parser)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the VGB CLI.

    Registered as the ``vgb`` console script in pyproject.toml.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "train": _handle_train,
        "eval": _handle_eval,
        "infer": _handle_infer,
        "export": _handle_export,
        "resume": _handle_resume,
        "inspect-checkpoint": _handle_inspect_checkpoint,
        "validate-config": _handle_validate_config,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()

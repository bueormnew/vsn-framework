"""Smoke test: CLI subcommands respond to --help without error.

Calls build_parser() and verifies each subcommand's --help
exits cleanly with code 0 and prints usage information.

Validates: Requirements 13.4
"""

from __future__ import annotations

import argparse

import pytest

from vgb.cli.main import build_parser


# All subcommands defined in the CLI
SUBCOMMANDS = [
    "train",
    "eval",
    "infer",
    "export",
    "resume",
    "inspect-checkpoint",
    "validate-config",
]


class TestCLISmoke:
    """Smoke tests for CLI --help on all subcommands."""

    def test_parser_builds_without_error(self):
        """build_parser() should not crash."""
        parser = build_parser()
        assert parser is not None
        assert parser.prog == "vgb"

    def test_top_level_help(self, capsys):
        """Top-level --help exits with code 0."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "vgb" in captured.out.lower()

    @pytest.mark.parametrize("subcommand", SUBCOMMANDS)
    def test_subcommand_help(self, subcommand: str, capsys):
        """Each subcommand responds to --help without crash."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([subcommand, "--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Should print some usage text
        assert len(captured.out) > 0

    @pytest.mark.parametrize("subcommand", SUBCOMMANDS)
    def test_subcommand_recognized(self, subcommand: str):
        """Parser recognizes each subcommand (doesn't error on parse)."""
        parser = build_parser()
        # All subcommands are registered in the subparsers
        # Verify by checking the _subparsers actions
        subparsers_actions = [
            action
            for action in parser._subparsers._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        assert len(subparsers_actions) > 0
        choices = subparsers_actions[0].choices
        assert subcommand in choices, (
            f"Subcommand '{subcommand}' not found in parser choices"
        )

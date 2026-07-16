"""Command-line interface: train, eval, infer, export, resume, inspect, validate.

Validates: Requirements 11.1
"""

from vgb.cli.main import main, build_parser

__all__ = ["main", "build_parser"]

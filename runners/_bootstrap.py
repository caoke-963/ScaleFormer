#!/usr/bin/env python
# coding=utf-8

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_project_root():
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_from_root(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return str(candidate)
    if len(path.parts) == 1:
        option_candidate = PROJECT_ROOT / 'options' / path
        if option_candidate.exists():
            return str(option_candidate)
    return str(candidate)

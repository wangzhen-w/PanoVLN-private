"""Load grouped YAML as CLI arguments so argparse validates both paths equally."""

from __future__ import annotations

import argparse
from pathlib import Path


CONFIG_SECTIONS = {"server", "navigation", "robot", "motion", "odometry", "camera", "recording", "upload"}


def config_arguments(parser: argparse.ArgumentParser, path: str) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("YAML configuration requires PyYAML; install it in the client Python environment") from exc
    try:
        config = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping of named sections")

    actions = {}
    for action in parser._actions:
        if action.dest not in {"help", "config", "print_config"}:
            actions.setdefault(action.dest, action)
    arguments = []
    seen = set()
    for section, values in config.items():
        if section not in CONFIG_SECTIONS or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        for key, value in values.items():
            if key not in actions:
                raise ValueError(f"Unknown configuration option: {section}.{key}")
            if key in seen:
                raise ValueError(f"Configuration option appears in multiple sections: {key}")
            seen.add(key)
            if value is None:
                continue
            action = actions[key]
            option = action.option_strings[0]
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    raise ValueError(f"{section}.{key} must be a YAML boolean (true/false)")
                if value:
                    arguments.append(option)
            elif isinstance(action.nargs, int):
                if not isinstance(value, list) or len(value) != action.nargs:
                    raise ValueError(f"{section}.{key} requires a list of {action.nargs} values")
                if any(isinstance(item, (bool, dict, list)) or item is None for item in value):
                    raise ValueError(f"Invalid list value for {section}.{key}")
                arguments.extend([option, *map(str, value)])
            else:
                if isinstance(value, (bool, dict, list)):
                    raise ValueError(f"{section}.{key} requires a scalar; quote string values such as 'yes'")
                # The '=' form also handles instructions that start with '-'.
                arguments.append(f"{option}={value}")
    return arguments

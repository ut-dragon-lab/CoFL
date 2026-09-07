"""Typed YAML and command-line recipes using jsonargparse."""

import json

from jsonargparse import ActionConfigFile, ArgumentParser

from .config import EvaluationConfig


def main(args=None):
    parser = ArgumentParser(prog="cofl evaluate", description="Batched offline policy evaluation")
    parser.add_argument("--config", action=ActionConfigFile, help="Evaluation recipe YAML or JSON")
    parser.add_class_arguments(EvaluationConfig)
    values = parser.instantiate(parser.parse_args(args)).as_dict()
    values.pop("config", None)
    try:
        config = EvaluationConfig(**values)
        from .runner import evaluate

        summary = evaluate(config)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    console = {
        key: value for key, value in summary.items() if key not in {"by_scene", "by_category"}
    }
    console["output"] = config.output
    print(json.dumps(console, indent=2, ensure_ascii=False, allow_nan=False))
    return 0

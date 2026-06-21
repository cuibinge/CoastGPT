import argparse
import logging
import os
from argparse import Namespace
from copy import deepcopy
from typing import List, Optional

import yaml

logger = logging.getLogger("train")


class ConfigArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.config_parser = argparse.ArgumentParser(add_help=False)
        self.config_parser.add_argument(
            "-c",
            "--config",
            default="Configs/step3_dual.yaml",
            metavar="FILE",
            help="where to load YAML configuration",
        )
        self.option_names = []
        super().__init__(
            *args,
            parents=[self.config_parser],
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **kwargs,
        )

    def add_argument(self, *args, **kwargs):
        arg = super().add_argument(*args, **kwargs)
        self.option_names.append(arg.dest)
        return arg

    def parse_args(self, wandb=False, args=None):
        res, remaining_argv = self.config_parser.parse_known_args(args)

        if res.config is not None:
            with open(res.config, "r") as f:
                config_vars = yaml.safe_load(f)
            namespace = vars(super().parse_args(remaining_argv))
            if wandb:
                config_vars.update({k: v for k, v in namespace.items() if v is not None})
                return config_vars
            else:
                namespace.update(config_vars)
                return namespace
        else:
            return vars(super().parse_args(remaining_argv))


def save_args(
    args: Namespace, filepath: str, excluded_fields: Optional[List[str]] = None
) -> None:
    """
    genericｆgeneric?YAML genericgenericgeneric€generic€genericgeneric?

    Args:
        args (Namespace): genericｆgeneric?
        filepath (str): generic』generic?".yaml" generic?
        excluded_fields (list[str]): generic€generic€genericgenericㄣ€?
            genericgeneric ["config"]generic?
    """
    assert isinstance(args, Namespace)
    assert filepath.endswith(".yaml")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    save_dict = deepcopy(args.__dict__)
    for field in excluded_fields or ["config"]:
        save_dict.pop(field)
    with open(filepath, "w") as f:
        yaml.dump(save_dict, f)
    logger.info(f"Args is saved to {filepath}")


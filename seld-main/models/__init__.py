import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key, value in config.items():
        if isinstance(value, str) and value and not os.path.isabs(value):
            config[key] = os.path.normpath(os.path.join(REPO_ROOT, value))
    return config

"""Experimental framework for baseline and proposed LWE secret generators."""

from .config import ExperimentConfig, build_default_config, load_experiment_config

__all__ = [
    "ExperimentConfig",
    "build_default_config",
    "load_experiment_config",
    "run_experiment_suite",
]


def __getattr__(name):
    if name == "run_experiment_suite":
        from .pipeline import run_experiment_suite

        return run_experiment_suite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

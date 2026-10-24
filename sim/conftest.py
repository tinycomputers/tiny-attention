"""Shared pytest/cocotb configuration: fixed seed so CI and local runs match."""

import random

import numpy as np

SEED = 42


def pytest_configure(config) -> None:
    random.seed(SEED)
    np.random.seed(SEED)

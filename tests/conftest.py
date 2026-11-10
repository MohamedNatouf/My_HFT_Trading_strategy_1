import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def random_walk_prices(symbols, n=900, seed=0, drift=0.0, vol=0.01, start="2015-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    r = drift + vol * rng.standard_normal((n, len(symbols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), idx, symbols)


@pytest.fixture
def rw():
    return random_walk_prices

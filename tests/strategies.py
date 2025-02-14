from hypothesis import settings
from hypothesis.strategies import floats, integers

import minitorch
import pytest
from hypothesis import strategies as st


settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
small_floats = st.floats(min_value=-100, max_value=100, allow_infinity=False, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


def assert_close(a, b):
    assert abs(a - b) < 1e-2

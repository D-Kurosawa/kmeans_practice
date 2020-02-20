import inspect

import pandas as pd
import pytest

from prac_kmeans.mypkg import pddisplay


@pytest.fixture()
def my_custom():
    obj = pd.options.display
    pddisplay.PandasDisplay.custom()

    customs = inspect.getmembers(obj, inspect.ismethod(obj))
    return {cst[0]: cst[1] for cst in customs}


def test_min_rows(my_custom):
    key = 'min_rows'
    assert my_custom[key] == pddisplay.PandasDisplay.custom_dict[key]


def test_max_rows(my_custom):
    key = 'max_rows'
    assert my_custom[key] == pddisplay.PandasDisplay.custom_dict[key]


def test_max_columns(my_custom):
    key = 'max_columns'
    assert my_custom[key] == pddisplay.PandasDisplay.custom_dict[key]


def test_width(my_custom):
    key = 'width'
    assert my_custom[key] == pddisplay.PandasDisplay.custom_dict[key]


def test_show_dimensions(my_custom):
    key = 'show_dimensions'
    assert my_custom[key] == pddisplay.PandasDisplay.custom_dict[key]


if __name__ == '__main__':
    pytest.main()

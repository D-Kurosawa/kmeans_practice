import tempfile
from pathlib import Path

import pytest

from prac_kmeans.mypkg import bz2pkl


@pytest.fixture()
def my_data():
    return dict(zip(range(1000000), range(1000000)))


def test_file_pickle(my_data):
    file_name = 'dump.pkl'

    with tempfile.TemporaryDirectory() as dir_name:
        file = Path(dir_name) / file_name
        bz2pkl.dump(my_data, file, compress=False)
        assert bz2pkl.load(file) == my_data


def test_file_pickle_bz2(my_data):
    file_name = 'dump.pkl.bz2'

    with tempfile.TemporaryDirectory() as dir_name:
        file = Path(dir_name) / file_name
        bz2pkl.dump(my_data, file)
        assert bz2pkl.load(file) == my_data


def test_memory_pickle(my_data):
    dumps = bz2pkl.dumps(my_data, compress=False)
    assert bz2pkl.loads(dumps) == my_data


def test_loads_and_dumps_pickle_bz2(my_data):
    dumps = bz2pkl.dumps(my_data)
    assert bz2pkl.loads(dumps) == my_data


if __name__ == '__main__':
    pytest.main()

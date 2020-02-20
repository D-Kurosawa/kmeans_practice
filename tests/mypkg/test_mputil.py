import pytest

from prac_kmeans.mypkg import mputil


def test_mpcpu():
    """Test machine CPU : Core i7-7700K (8core)"""
    data = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
        (0, 1), (-10, 1), (10, 8), (None, 7),
        ('max', 8), ('min', 1), ('mid', 4), ('auto', 7),
        ('1/4', 2), ('1/2', 4), ('3/4', 6)
    ]

    for cpu, rst in data:
        obj = mputil.MpCPU(cpu)
        assert obj.get() == rst


def test_mpcpu_exceptions():
    """TypeError raises Pycharm inspection error"""
    with pytest.raises(KeyError):
        mputil.MpCPU('1/5').get()

    # with pytest.raises(TypeError):
    #     mputil.MpCPU(2.0).get()
    #
    # with pytest.raises(TypeError):
    #     mputil.MpCPU([1, 2]).get()


def test_mpcounter():
    num = 100

    for _ in range(num):
        mputil.MpCounter()

    assert mputil.MpCounter().num == num + 1

    mputil.MpCounter().count_reset()
    assert mputil.MpCounter().num == 1


def test_mplines(capsys):
    obj = mputil.MpLines()

    obj.top()
    out, err = capsys.readouterr()
    assert out == f"\n{'*' * 60}\n<Multi process Start>\n{'*' * 60}\n"

    obj.bottom()
    out, err = capsys.readouterr()
    assert out == f"{'*' * 60}\n<Multi process End>\n{'*' * 60}\n\n"


if __name__ == '__main__':
    pytest.main()

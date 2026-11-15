# lint-as: tests/unit/test_example.py
def test_real(x=1):
    assert x == 1
    assert x, "nonzero"
    assert f() == f()   # calls may differ; not reported


def f():
    return 1

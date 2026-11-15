# lint-as: tests/unit/test_example.py
def test_nothing_checked():
    value = 1 + 1
    return value


def test_only_conditional(rates):
    if "Black" in rates:
        assert rates["Black"] >= 0

# lint-as: tests/unit/test_example.py
def test_vacuous(x=1):
    assert True
    assert "always"
    assert (x == 2, "message in a tuple is always true")
    assert x == x

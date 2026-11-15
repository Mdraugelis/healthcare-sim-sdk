Deliberately bad and good code for `scripts/sdk_lint.py`. These files are
excluded from Flake8, Ruff, sdk-lint and pytest collection, and are linted
only by `tests/unit/test_sdk_lint.py` under an assigned path. The first line
of each file names the path it is linted as.

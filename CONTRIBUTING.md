# Contributing

Use Python 3.12+ and work from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
python -m twine check dist/*
```

For the pinned numerical dependency versions, pass
`-c requirements-validation.txt` to the install command. The package metadata
uses compatible ranges; the constraints describe the validated environment.

Keep economic equations separate from the Newton backend. Preserve importer /
exporter / sector axes, closure, normalization, and exact welfare definitions.
Do not silently reconcile baselines or replace zero trade with positive values.
Changes to model conventions require documentation and reference validation.

Tests construct synthetic baselines analytically, compare independent
formulations, and check conservation and derivative identities. Maintain these
checks when changing the model; do not treat agreement between two solvers as
a substitute for an analytic or economic check.

Before release, build a wheel, install it into a fresh environment, change out
of the checkout, and run the examples and tests against the installed package.
CI includes this check. Test the actual wheel, not just editable imports.

The initial package version is in `pyproject.toml` and
`src/cpmodel_jax/__init__.py`; a test checks they agree. Update both, the
changelog, and `CITATION.cff` for releases. No automatic publishing workflow is
configured. Choose an available PyPI name and add verified repository URLs
before publishing. Benchmark cold and warm calls separately and retain raw
samples and state the measured hardware, precision, and compilation policy.

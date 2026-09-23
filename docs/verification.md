# Local verification — version 0.4.0

Validated on 2026-09-23 with macOS arm64, Python 3.12.14, JAX/jaxlib 0.11.2,
NumPy 2.5.3, CPU, float64.

- Ruff lint and formatting: passed.
- Standalone test suite using descriptive input names: 23 passed.
- Installed wheel in an isolated environment outside the checkout: 23 passed.
- Wheel/source builds and strict distribution metadata checks: passed.
- Custom-input example also runs from the installed wheel.
- All 18 outputs are bitwise identical to version 0.3.0 for seven existing toy
  scenarios and both formulations (14 scenario/formulation pairs).
- The complete custom-input example runs with all six calibration fields and
  all three policy arguments. Its maximum economic equation error is below
  `1e-8`.
- Source audit found no remaining symbolic input names in Python code, tests,
  or examples. Mathematical notation in the method guide remains conventional.

Version 0.4 changes input keywords and stored calibration attributes. The
sample data, output names, economic equations, and numerical algorithm are
unchanged. Test coverage includes analytic equilibria, formulation agreement,
accounting, directional derivatives, units, country permutation, zero links,
input validation, and numerical failure handling.

GitHub Actions is configured but has not run remotely. The package has not
been published to GitHub or PyPI.

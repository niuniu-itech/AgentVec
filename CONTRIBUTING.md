# Contributing

Install `python -m pip install -e '.[dev,remote]'`, then run `python -m pytest`.
Keep model proposals separate from caller-owned legality rules. A static check
produces CHECKED, and a successful independent differential test produces VERIFIED.
Changes to lowering need tests for rejected contracts and vector tails, as well as
a native or emulated compilation check when the toolchain is available.

Keep measurement outputs in an external run directory. Do not edit the recorded
`results/rvv` artifacts to match a new implementation or a desired performance number.
Describe which tests were actually run and identify the compiler and target.

Use concise module comments to explain contracts or non-obvious numerical choices.
Submit small pull requests with the behavior change and validation commands.

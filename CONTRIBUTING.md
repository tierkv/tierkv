# Contributing to tierkv

## Development setup

**Python layer** (CLI, EXO hook, config):

```bash
uv sync
pip install -e .
```

**Rust crate** (TurboQuant, gRPC vault, tiered cache):

```bash
cd tierkv-core
maturin develop --release
```

After changing Rust code, re-run `maturin develop` to rebuild the extension module.

## Making changes

- One feature or fix per PR.
- Describe the hardware you tested on (model, RAM, network topology) — behavior differs significantly between single-machine and multi-node setups.
- If you change the gRPC protocol (`proto/`), regenerate the stubs: `python -m grpc_tools.protoc ...` and commit the generated files.

## Reporting issues

Include: tierkv version, EXO version (from `pip show exo` or the EXO git SHA), OS, and the relevant section of `/tmp/tierkv.log`.

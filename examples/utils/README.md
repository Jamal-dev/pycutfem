# `examples/utils` package

Helpers in this folder are **example-domain code**, not core library API.

- `biofilm/`: biofilm one-domain forms, adhesion, and MMS helpers.
- `fsi/`: fully Eulerian FSI benchmark helpers.
- `fpi/`: fluid-poroelastic interaction helpers and MMS utilities.
- `shared/`: shared building blocks used by the example-domain modules.
- `debug/`: debug/investigation helpers.

If a helper becomes broadly reusable across problems, promote it into
`pycutfem/` with dedicated tests and a stable API.


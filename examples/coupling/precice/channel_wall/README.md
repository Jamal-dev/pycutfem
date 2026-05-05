# preCICE Channel-Wall Demo

This directory contains the first end-to-end `pycutfem` / `fenicsxprecice` coupling added for this repository:

- `pycutfem_fluid_participant.py`: `pycutfem` Stokes channel participant using the generic `pyprecice` point-cloud wrapper in `pycutfem/coupling/precice.py`.
- `fenicsx_solid_participant.py`: FEniCSx elastic wall participant using the official `fenicsxprecice` adapter.
- `precice-config.xml`: serial-implicit preCICE coupling with `IQN-ILS` acceleration.
- `solid-precice-adapter-config.json`: adapter-side JSON config for the solid participant.
- `run_demo.py`: launches both participants in the `fenicsx` conda environment.

The coupling model is intentionally small and robust:

- The fluid participant solves a channel Stokes problem on `Omega_f = [0,1] x [0,1]`.
- The solid participant solves a linear-elastic wall problem on `Omega_s = [0,1] x [1,1.25]`.
- The exchanged data are scalar interface fields:
  - `Pressure` from fluid to solid.
  - `DisplacementY` from solid to fluid.

This is the first external preCICE split. The one-domain Seboldt benchmark remains an internal staggered fixed-point solve because its current flow stage still depends on full porous/solid fields, not only interface data.

Run:

```bash
conda run --no-capture-output -n fenicsx python examples/coupling/precice/channel_wall/run_demo.py
```

Validate the preCICE XML first if needed:

```bash
conda run --no-capture-output -n fenicsx precice-config-validate examples/coupling/precice/channel_wall/precice-config.xml
```

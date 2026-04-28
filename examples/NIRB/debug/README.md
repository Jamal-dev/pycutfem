# NIRB Debug Utilities

This folder contains one-off scripts and notes used to compare pycutfem against
Kratos while closing the DoubleFlap/NIRB parity issue.

The production DVMS operator implementation remains in `examples/NIRB/dvms/`.
The files here are intentionally kept out of the main NIRB driver path; use them
only for audits, dumps, and regression investigations against Kratos artifacts.

# Psi4 on native Windows: status probe

**Date:** 2026-07-04 · **Command:** `conda search -c conda-forge psi4 --platform win-64`

## Result

Psi4 has native Windows builds on conda-forge. The probe came back with 134 package records covering 1.8, 1.8.1, 1.8.2, 1.9, 1.9.1, 1.10, 1.10.1, 1.10.2 and 1.11. Newest is **1.11**, built for Python 3.10 through 3.14. Everything is on the conda-forge channel.

## Decision

Psi4 1.11 runs natively on Windows, so the validation cross-checks go ahead as the Hartree-Fock phase specced them. Nothing to do now. Psi4 stays out of the Phase 0/1 environment on purpose.

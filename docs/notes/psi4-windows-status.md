# Psi4 on native Windows: status probe

**Date:** 2026-07-04 · **Command:** `conda search -c conda-forge psi4 --platform win-64`

## Result

Psi4 has native Windows builds available on conda-forge. The probe returned 134 package records spanning versions 1.8, 1.8.1, 1.8.2, 1.9, 1.9.1, 1.10, 1.10.1, 1.10.2, and 1.11. The newest available version is **1.11**, with builds for Python 3.10, 3.11, 3.12, 3.13, and 3.14. All builds are hosted on the conda-forge channel.

## Decision

Psi4 1.11 is available natively on Windows. Validation cross-checks proceed as specced in the Hartree-Fock phase. No action now, Psi4 is deliberately NOT part of the Phase 0/1 environment.

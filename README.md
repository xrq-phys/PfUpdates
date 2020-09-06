This is a simple test for N-electron Pfaffian update performance.
At the moment, PfUpdate_N.cc is the only entry. Just compile PfUpdate_N.cc with appropriate compilers should do the job.

- This test is single-threaded.
- Constant `n_test` and `n_update` controls number of tests and update interval (i.e. "N" in the "N-electron" update).

Roadmaps:
- Test hopping-without-updating performance.
- Avoid in-routine allocation.
- Instead of GETRI/GETRF & SKPFA, use tridiagonal factorization.
- Use SKR2K assembly.
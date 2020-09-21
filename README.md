This is a simple test for *m*-electron Pfaffian update performance.
At the moment, `PfUpdate_Tests.cc` is the only entry. Just compile `PfUpdate_Tests.cc` with appropriate compilers should do the job.

- This test is single-threaded.
- Constant `n_test` and `n_update` controls number of tests and update interval (i.e. "*m*" in the "*m*-electron" update).
- Sorry but inside the program variable `k` is used instead of *m* to denote this size of update.

Roadmaps:
- [ ] Test hopping-without-updating performance.
- [x] Avoid in-routine allocation.
- [x] Instead of GETRI/GETRF & SKPFA, use tridiagonal factorization.
- [x] Use SKR2K assembly.

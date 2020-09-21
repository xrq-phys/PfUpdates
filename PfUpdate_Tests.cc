// PfUpdate_N.cc : 
//   This file contains the 'main' function. 
//   Program execution begins and ends there.
//   A simple timing test for m-electron updates.
//

#include "orbital_mat.tcc"
#include "updated_tdi.tcc"
#include <iostream>
#include <chrono>
#include <random>
#ifdef _Intel_Advisor
#include <ittnotify.h>
#endif

int main(const int argc, const char *argv[]) {
  using namespace std;
  using namespace std::chrono;

#ifdef _Intel_Advisor
  __itt_pause();
#endif

#ifdef _DEBUG
  const dim_t nsite = 40;
#else
  const dim_t nsite = 450;
#endif
  const dim_t npar = nsite * nsite;
  double *Xpar = new double[npar];
  orbital_mat<double> param(BLIS_UPPER, nsite, Xpar, signed(nsite));
  param.randomize(2e-1, 511);

#ifdef _DEBUG
  const dim_t nfermi = 20;
#else
  const dim_t nfermi = 400;
#endif
  const dim_t nmat = nfermi * nfermi;
  double *Mbuf = new double[nmat];
#ifdef _DEBUG
  double *Mbuf_v = new double[nmat];
#endif
  vector<dim_t> cfg(nfermi, 0);
  vector<bool> mark(nsite, true);

  // Stupid shuffling.
  mt19937_64 rng(511);
  uniform_int_distribution<int> dist(0);
  dim_t nempty = nsite;
  for (int i = 0; i < nfermi; ++i) {
    int pos = dist(rng) % nempty;
    int cpos = 0;

    while (!mark.at(cpos)) ++cpos;
    for (; pos != 0; --pos) {
      while (!mark.at(cpos))
        ++cpos;
      ++cpos; 
    }
    while (!mark.at(cpos)) ++cpos;

    mark.at(cpos) = false;
    cfg.at(i) = cpos;
    --nempty;
  }

  const dim_t ksize = 32;
  updated_tdi<double> Xij(param, cfg, Mbuf, nfermi, ksize);

  const dim_t n_update = 8;
#ifdef _DEBUG
  const dim_t n_test = 4;
#else
  const dim_t n_test = 2000;
#endif

  auto start = high_resolution_clock::now();
#ifdef _Intel_Advisor
  __itt_resume();
#endif
  
  for (dim_t itest = 0; itest < n_test; ++itest) {
    for (dim_t i = 0; i < n_update; ++i) {
      // Get update.
      dim_t msj = dist(rng) % nfermi;
      dim_t osj = Xij.elem_cfg.at(msj);
      int pos = dist(rng) % nempty;
      int cpos = 0;

      // Check if already hopped out.
      for (dim_t l = 0; l < Xij.from_idx.size(); ++l)
        if (msj == Xij.from_idx.at(l))
          goto NEXT_LOOP;

      while (!mark.at(cpos)) ++cpos;
      for (; pos != 0; --pos) {
        while (!mark.at(cpos))
          ++cpos;
        ++cpos;
      }
      while (!mark.at(cpos)) ++cpos;

      // Check for hopping back in.
      for (dim_t l = 0; l < Xij.to_site.size(); ++l)
        if (cpos == Xij.to_site.at(l))
          goto NEXT_LOOP;

      if (cpos != osj) {
        mark.at(cpos) = false;
        mark.at(osj) = true;
        Xij.push_update(cpos, msj);
      }
    NEXT_LOOP:
      cpos = 0;
    }
#ifdef _DEBUG
    cout << "info: k=" << Xij.from_idx.size();
#endif
    Xij.merge_updates();
#ifdef _DEBUG
    cout << " merging. Pfa=" << Xij.get_Pfa() << endl;
    updated_tdi<double> Xij_v(param, Xij.elem_cfg, Mbuf_v, nfermi, 1);
    cout << " Verifying Pfa=" << Xij_v.get_Pfa() << endl;
    double difA = 0.0;
    for (dim_t j = 0; j < nfermi; ++j)
      for (dim_t i = 0; i < nfermi; ++i)
        difA += abs(Xij.M(i, j) - Xij_v.M(i, j)) / nfermi / nfermi;
    cout << " Diff of inv=" << difA << endl;
#endif
  }

  auto elapsed = high_resolution_clock::now() - start;
  long long mus = duration_cast<microseconds>(elapsed).count();
  double mis = double(mus) / 1000;
#ifdef _Intel_Advisor
  __itt_pause();
#endif

  cout << "Test for n=" << nfermi << " with " << n_update
       << " electron update elapsed " << mis << " ms." << endl
       << "Time per evaluation is " << mis / n_test << " ms." << endl
       << "Time per update is " << mis / n_test / n_update << " ms. " << endl;

  delete[] Xpar;
  delete[] Mbuf;
#ifdef _DEBUG
  delete[] Mbuf_v;
#endif

  return 0;
}

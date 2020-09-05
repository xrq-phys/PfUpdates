// PfUpdate_N.cc : This file contains the 'main' function. Program execution begins and ends there.
//

#include "inv_update_n.tcc"
#include <iostream>
#include <chrono>
#include <random>

int main(const int argc, const char *argv[]) {
  using namespace std;
  using namespace std::chrono;

  const unsigned nsite = 450;
  const unsigned npar = nsite * nsite;
  double *Xpar = new double[npar];
  orbital_Xij<double> param(nsite, Xpar, signed(nsite));
  param.randomize(511);

  const unsigned nfermi = 400;
  const unsigned nmat = nfermi * nfermi;
  double *Mbuf = new double[nmat];
  vector<int> cfg(nfermi, 0);
  vector<bool> mark(nsite, true);

  // Stupid shuffling.
  mt19937_64 rng(511);
  uniform_int_distribution<int> dist(0);
  unsigned nempty = nsite;
  for (int i = 0; i < nfermi; ++i) {
    int pos = dist(rng) % nempty;
    int cpos = 0;

    while (!mark.at(cpos)) ++cpos;
    for (; pos != 0; --pos) {
      while (!mark.at(cpos))
        ++cpos;
      ++cpos; 
    }
    mark.at(cpos) = false;
    cfg.at(i) = cpos;
    --nempty;
  }

  const unsigned ksize = 32;
  double *Ubuf = new double[nfermi * ksize];
  double *Qbuf = new double[nfermi * ksize];
  double *Pbuf = new double[nfermi * ksize];
  double *Wbuf = new double[ksize * ksize * 4];
  double *Bbuf1 = new double[ksize * ksize * 4];
  double *Bbuf2 = new double[ksize * ksize * 4];
  double *Bbuf3 = new double[ksize * ksize * 4];
  updated_Xij<double> Xij(param, cfg, Mbuf, nfermi, Ubuf, Pbuf, Qbuf, Wbuf,
                          ksize * 2, Bbuf1, Bbuf2, Bbuf3);

  const unsigned n_update = 1;
  const unsigned n_test = 20;

  auto start = high_resolution_clock::now();
  
  for (int itest = 0; itest < n_test; ++itest) {
    for (int i = 0; i < n_update; ++i) {
      // Get update.
      int msj = dist(rng) % nfermi;
      int osj = Xij.elem_cfg.at(msj);
      int pos = dist(rng) % nempty;
      int cpos = 0;

      while (!mark.at(cpos)) ++cpos;
      for (; pos != 0; --pos) {
        while (!mark.at(cpos))
          ++cpos;
        ++cpos;
      }
      if (cpos != osj) {
        mark.at(cpos) = false;
        mark.at(osj) = true;
        push_Xij_update<double>(Xij, cpos, msj);
      }
    }
		apply_Xij_update<double>(Xij);
  }

  auto elapsed = high_resolution_clock::now() - start;
  long long mus = duration_cast<microseconds>(elapsed).count();
  double mis = double(mus) / 1000;

  cout << "Test for n=" << nfermi << " with " << n_update
       << " electron update elapsed " << mis << " ms." << endl
       << "Time per evaluation is " << mis / n_test << " ms." << endl
       << "Time per update is " << mis / n_test / n_update << " ms. " << endl;

  delete[] Xpar;
  delete[] Mbuf;
  delete[] Ubuf;
  delete[] Qbuf;
  delete[] Pbuf;
  delete[] Wbuf;
  delete[] Bbuf1;
  delete[] Bbuf2;
  delete[] Bbuf3;

  return 0;
}

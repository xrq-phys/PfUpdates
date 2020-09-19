#include "colmaj.tcc"
#include "blalink.hh"
#include "skpfa.hh"
#include "blis.h"
#include <vector>
#include <random>
#include <iostream>


template <typename T>
struct orbital_Xij
{
  const unsigned nsite;
  const int ldX;
  T *X_; ///< nsite*nsite.

  orbital_Xij(unsigned nsite_, T *X, int ldX_)
  : nsite(nsite_), X_(X), ldX(ldX_) { }

  orbital_Xij(unsigned nsite_, colmaj<T> &X)
  : nsite(nsite_), X_(X.dat), ldX(X.ld) { }

  void randomize(unsigned seed) { 
    using namespace std;
    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(-0.1, 1.0);

    colmaj<T> X(X_, ldX);
    for (unsigned j = 0; j < nsite; ++j)
      for (unsigned i = 0; i < j; ++i) {
        X(i, j) = T(dist(rng));
        X(i, j) = -X(j, i);
      }
  }

  void randomize() { randomize(511); }
};

template <typename T>
struct updated_Xij
{
  orbital_Xij<T> &param;

  const unsigned n;
  const int ldM;
  T *M_; ///< n*n.
  T *U_, *P_, *Q_;
  const int ldW;
  T *W_;
  T *UMU_, *VMV_, *UMV_;
  T Pfa;

  std::vector<int> elem_cfg;
  std::vector<int> from_idx;
  std::vector<int> to_site;

  updated_Xij(orbital_Xij<T> param_, std::vector<int> cfg, 
              T *M, int ldM_, T *U, T *P, T *Q, 
              T *W, int ldW_, T *UMU, T *VMV, T *UMV)
      : param(param_), n(cfg.size()), M_(M), ldM(ldM_), U_(U), P_(P), Q_(Q), W_(W), ldW(ldW_),
        UMU_(UMU), VMV_(VMV), UMV_(UMV), elem_cfg(cfg), from_idx(0), to_site(0), Pfa(0.0) { 
    using namespace std;
    colmaj<T> A(M_, ldM);
    colmaj<T> X(param.X_, param.ldX);
    colmaj<T> G(new T[n * n], n);
     
    for (unsigned j = 0; j < n; ++j) {
      for (unsigned i = 0; i < j; ++i) {
        A(i, j) = X(cfg.at(i), cfg.at(j));
        A(j, i) = -A(i, j);
      }
      A(j, j) = T(0.0);
    }
    
    // TODO: Compute Pfaffian.
    // Allocate scratchpad.
    signed *iPov = new signed[n + 1];
    signed lwork = n * 16; ///< npanel = 16 is just a easy value.
    T *pfwork = new T[lwork];

    signed info = skpfa(BLIS_UPPER, n, &A(0, 0), A.ld, &G(0, 0), G.ld, iPov,
                        true, &Pfa, pfwork, lwork);
#ifdef _DEBUG
    cout << "SKPFA+INV: n=" << n << " info=" << info << endl;
#endif

    delete[] iPov;
    delete[] pfwork;
    delete[](&G(0, 0));
  }
};

template <typename T>
void push_Xij_update(updated_Xij<T> &Xij, int osi, int msj, bool compute_pfaff) {
  using namespace std;

  // This is the k-th hopping.
  int n = Xij.n;
  int k = Xij.from_idx.size();
  int osj = Xij.elem_cfg.at(msj);

  // TODO: Check?
  Xij.from_idx.push_back(msj);
  Xij.to_site.push_back(osi);

  colmaj<T> U(Xij.U_, Xij.ldM);
  colmaj<T> P(Xij.P_, Xij.ldM);
  colmaj<T> Q(Xij.Q_, Xij.ldM);
  colmaj<T> M(Xij.M_, Xij.ldM);
  colmaj<T> Xpar(Xij.param.X_, Xij.param.ldX);
  colmaj<T> W(Xij.W_, Xij.ldW);
  colmaj<T> UMU(Xij.UMU_, Xij.ldW);
  colmaj<T> VMV(Xij.VMV_, Xij.ldW);
  colmaj<T> UMV(Xij.UMV_, Xij.ldW);

  for (int i = 0; i < n; ++i) {
    U(i, k) = Xpar(Xij.elem_cfg.at(i), osi) - Xpar(Xij.elem_cfg.at(i), osj);
    P(i, k) = M(i, msj);
  }
  gemv(BLIS_NO_TRANSPOSE, n, n, T(1.0), &M(0, 0), Xij.ldM, &U(0, k), 1, T(0.0),
       &Q(0, k), 1);

  for (int l = 0; l < k; ++l)
    W(l, k) = -Xpar(Xij.to_site.at(l), osi) + Xpar(Xij.to_site.at(l), osj);

  for (int l = 0; l < k; ++l)
    UMU(l, k) = dot(n, &U(0, l), 1, &Q(0, k), 1);
  for (int l = 0; l < k; ++l) {
    UMV(l, k) = dot(n, &U(0, l), 1, &P(0, k), 1);
    UMV(k, l) = dot(n, &U(0, k), 1, &P(0, l), 1);
  }
  UMV(k, k) = dot(n, &U(0, k), 1, &P(0, k), 1);
  for (int l = 0; l < k; ++l)
    VMV(l, k) /*= -VMV(k, l)*/ = -P(msj, l); ///< P(Xij.from_idx.at(l), k);

  // If it's the first update, C is not needed.
  if (k == 0) {
    if (compute_pfaff)
      Xij.Pfa *= -UMV(0, 0);
    return;
  }

  if (compute_pfaff) {
    // NOTE: Refresh k to be new size.
    k += 1;
    // Allocate scratchpad.
    colmaj<T> SpG(new T[2 * k * 2 * k], 2 * k);
    signed *iPov = new signed[2 * k + 1];
    signed lwork = 2 * k * 2 * k;
    T *pfwork = new T[lwork];

    // Assemble upper half of C+BMB buffer.
    colmaj<T> C(new double[2 * k * 2 * k], 2 * k);
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < j; ++i)
        C(i, j) = W(i, j) + UMU(i, j);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < k; ++i)
        C(i, j + k) = UMV(i, j);

      C(j, j + k) -= T(1.0);
      C(j + k, j) -= T(-1.0);
    }
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < j; ++i)
        C(i, j) = VMV(i, j);

    double PfaMul;
    // Compute pfaffian.
    signed info = skpfa(BLIS_UPPER, 2 * k, &C(0, 0), 2 * k, &SpG(0, 0), 2 * k,
                        iPov, false, &PfaMul, pfwork, lwork);
#ifdef _DEBUG
    cout << "SKPFA: info=" << info << endl;
#endif
    if (k % 2)
      // The 1st, 3rd, 5th... update in the series.
      Xij.Pfa *= -PfaMul;
    else
      Xij.Pfa *= PfaMul;

    delete[] iPov;
    delete[] pfwork;
    delete[](&SpG(0, 0));
    delete[](&C(0, 0));
  } else
    // Set to 0.0 and awaits computation @ update time.
    Xij.Pfa = 0.0;

}

template <typename T>
void push_Xij_update(updated_Xij<T> &Xij, int osi, int msj) {
  push_Xij_update(Xij, osi, msj, true);
}

/**
 * K-update of inverse matrix.
 */
template <typename T>
void inv_update_n(unsigned n, unsigned k, T *A_, int ldA, T *U_, T *Q_, T *P_,
                  T *D_, T *E_, T *F_, int ldC) {
  // Now k stands for cnt of total hopping terms.

  colmaj<T> A(A_, ldA);
  colmaj<T> U(U_, ldA);
  colmaj<T> Q(Q_, ldA);
  colmaj<T> P(P_, ldA);

  colmaj<T> D(D_, ldC);
  colmaj<T> E(E_, ldC);
  colmaj<T> F(F_, ldC);

  // BLAS scratchpads.
  colmaj<T> AUD(&U(0, 0), ldA); ///< Use U as AUD buffer.
  colmaj<T> AUF(new T[n * k], n);
  colmaj<T> AVE(new T[n * k], n);

  // NOTE: To do this UU/VV sectors, D and E must be complete antisymmetric instead of uplo stored.
  gemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, k, k, T(1.0), &Q(0, 0), ldA,
       &D(0, 0), ldC, T(0.0), &AUD(0, 0), ldA); ///< inv(A)UD
  gemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, k, k, T(1.0), &P(0, 0), ldA,
       &E(0, 0), ldC, T(0.0), &AVE(0, 0), n); ///< inv(A)VE
  
  gemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, k, k, T(1.0), &Q(0, 0), ldA,
       &F(0, 0), ldC, T(0.0), &AUF(0, 0), n); ///< inv(A)UF

  for (unsigned j = 0; j < n; ++j) {
    for (unsigned l = 0; l < k; ++l) {
      T Q_jk = Q(j, k);
      T P_jk = P(j, k);
      T AUF_jk = AUF(j, k);
    
      for (unsigned i = 0; i < n; ++i)
        A(i, j) += AUD(i, k) * Q_jk + P(i, k) * AUF_jk -
                   (AUF(i, k) + AVE(i, k)) * P_jk;
    }
  }

  // delete[](&AUD(0, 0));
  delete[](&AUF(0, 0));
  delete[](&AVE(0, 0));
}

template <typename T>
void complete_antisym(const uplo_t uplo, unsigned n, T *A_, int ldA) {
  colmaj<T> A(A_, ldA);

  switch (uplo) {
  case BLIS_UPPER:
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < j; ++i)
        A(j, i) = -A(i, j);
    break;

  case BLIS_LOWER:
    for (int j = 0; j < n; ++j)
      for (int i = j + 1; i < n; ++i)
        A(j, i) = -A(i, j);
    break;

  default:
    return;
  }
}

template <typename T> void apply_Xij_update(updated_Xij<double> &Xij) {
  using namespace std;

  int n = Xij.n;
  int k = Xij.from_idx.size();
  if (k == 0)
    return;

  colmaj<T> W(Xij.W_, Xij.ldW);
  colmaj<T> UMU(Xij.UMU_, Xij.ldW);
  colmaj<T> VMV(Xij.VMV_, Xij.ldW);
  colmaj<T> UMV(Xij.UMV_, Xij.ldW);
  colmaj<T> G(new T[2 * k * 2 * k], 2 * k);

  // Repeat: assemble C+BMB buffer.
  colmaj<T> C(new double[2 * k * 2 * k], 2 * k);
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < j; ++i) {
      C(i, j) = W(i, j) + UMU(i, j);
      C(j, i) = -C(i, j);
    }
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i < k; ++i) {
      C(i, j + k) = UMV(i, j);
      C(j + k, i) = -UMV(i, j);
    }

    C(j, j + k) -= T(1.0);
    C(j + k, j) -= T(-1.0);
  }
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < j; ++i) {
      C(i, j) = VMV(i, j);
      C(j, i) = -VMV(i, j);
    }

  // Allocate scratchpad.
  signed *iPov = new signed[2 * k + 1];
  signed lwork = 2 * k * 2 * k;
  T *pfwork = new T[lwork];
  T PfaRatio;

  if (k == 1) {
    C(0, 1) = -1.0 / C(0, 1);
    C(1, 0) = -1.0 / C(1, 0); 
  } else {
    signed info = skpfa(BLIS_UPPER, 2 * k, &C(0, 0), C.ld, &G(0, 0), G.ld, iPov,
                        true, &PfaRatio, pfwork, lwork);
#ifdef _DEBUG
    cout << "SKPFA+INV: n=" << 2 * k << " info=" << info << endl;
#endif
  }
  inv_update_n<T>(n, k, Xij.M_, Xij.ldM, Xij.U_, Xij.Q_, Xij.P_, &C(0, 0),
                  &C(0, k), &C(k, k), 2 * k);

  // Apply hopping.
  for (int j = 0; j < k; ++j)
    Xij.elem_cfg.at(Xij.from_idx.at(j)) = Xij.to_site.at(j);
  Xij.from_idx.clear();
  Xij.to_site.clear();

  delete[](&G(0, 0));
  delete[](&C(0, 0));
  delete[] pfwork;
  delete[] iPov;
}

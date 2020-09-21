/**
 * \copyright Copyright (c) Dept. Phys., Univ. Tokyo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "orbital_mat.tcc"
#include "blalink.hh"
#include "skpfa.hh"
#include "sktdi.hh"
#include "skr2k.tcc"
#include <iostream>
#include <vector>

const bool single_hop_alpha = true;
const dim_t min_inv_update_blk = 4;
const dim_t npanel_big = 16;
const dim_t npanel_sub = 4;

template <typename T> struct updated_tdi {
  orbital_mat<T> &Xij;
  uplo_t uplo; ///< Can be switched only with update_uplo().
  const dim_t nelec;
  const dim_t mmax;

  // Matrix M.
  colmaj<T> M;

  // Updator blocks U, inv(M)U and inv(M)V.
  // Note that V is not stored.
  colmaj<T> U; ///< Serves also as B*inv
  colmaj<T> Q; ///< Contains the whole B buffer.
  colmaj<T> P; ///< Q(0, mmax)

  // Central update buffer and its blocks.
  colmaj<T> W;
  colmaj<T> UMU, UMV, VMV;
  colmaj<T> Cp;       ///< C+BMB = [ W -I; I 0 ] + [ UMU UMV; -UMV VMV ]
  colmaj<T> Gc;       ///< Gaussian vectors when tri-diagonalizing C.
  signed *const cPov; ///< Pivot when tri-diagonalizing C.
  T Pfa;
  T PfaRatio; //< Updated Pfaffian / Base Pfaffian.
  // TODO: Maybe one should log all Pfaffian histories.

  std::vector<dim_t> elem_cfg;
  std::vector<dim_t> from_idx;
  std::vector<dim_t> to_site;

  void initialize() {
    using namespace std;
    auto &cfg = elem_cfg;
    colmaj<T> G(new T[nelec * nelec], nelec);

    switch (uplo) {
    case BLIS_UPPER:
      for (dim_t j = 0; j < nelec; ++j) {
        for (dim_t i = 0; i < j; ++i) {
          M(i, j) = Xij(cfg.at(i), cfg.at(j));
        }
        M(j, j) = T(0.0);
      }
      break;

    default:
      cerr << "updated_tdi<T>: BLIS_LOWER is not supported. The error is fetal."
           << endl;
    }

    // Allocate scratchpad.
    signed *iPovFull = new signed[nelec + 1];
    dim_t lwork = nelec * npanel_big;
    T *pfwork = new T[lwork];

    signed info = skpfa(uplo, nelec, &M(0, 0), M.ld, &G(0, 0), G.ld, iPovFull,
                        true, &Pfa, pfwork, lwork);
#ifdef _DEBUG
    cout << "SKPFA+INV: n=" << nelec << " info=" << info << endl;
#endif

    delete[] iPovFull;
    delete[] pfwork;
    delete[](&G(0, 0));
  }

  ~updated_tdi() {
    delete[](&U(0, 0));
    delete[](&Q(0, 0));

    delete[](&Cp(0, 0));
    delete[](&Gc(0, 0));
    delete[] cPov;

    delete[](&W(0, 0));
    delete[](&UMU(0, 0));
    delete[](&UMV(0, 0));
    delete[](&VMV(0, 0));
  }

  updated_tdi(orbital_mat<T> &Xij_, std::vector<dim_t> &cfg, T *M_, inc_t ldM,
              dim_t mmax_)
      : Xij(Xij_), nelec(cfg.size()), mmax(mmax_), M(M_, ldM),
        U(new T[nelec * mmax * 2], nelec), Q(new T[nelec * mmax * 2], nelec),
        P(&Q(0, mmax), Q.ld), W(new T[mmax * mmax], mmax),
        UMU(new T[mmax * mmax], mmax), UMV(new T[mmax * mmax], mmax),
        VMV(new T[mmax * mmax], mmax), Cp(new T[2 * mmax * 2 * mmax], 2 * mmax),
        Gc(new T[2 * mmax * 2 * mmax], 2 * mmax),
        cPov(new signed[2 * mmax + 1]), Pfa(0.0), PfaRatio(1.0), elem_cfg(cfg),
        from_idx(0), to_site(0), uplo(BLIS_UPPER) {
    initialize();
  }

  updated_tdi(orbital_mat<T> &Xij_, std::vector<dim_t> &cfg, colmaj<T> &M_,
              dim_t mmax_)
      : Xij(Xij_), nelec(cfg.size()), mmax(mmax_), M(M_),
        U(new T[nelec * mmax * 2], nelec), Q(new T[nelec * mmax * 2], nelec),
        P(&Q(0, mmax), Q.ld), W(new T[mmax * mmax], mmax),
        UMU(new T[mmax * mmax], mmax), UMV(new T[mmax * mmax], mmax),
        VMV(new T[mmax * mmax], mmax), Cp(new T[2 * mmax * 2 * mmax], 2 * mmax),
        Gc(new T[2 * mmax * 2 * mmax], 2 * mmax),
        cPov(new signed[2 * mmax + 1]), Pfa(0.0), PfaRatio(1.0), elem_cfg(cfg),
        from_idx(0), to_site(0), uplo(BLIS_UPPER) {
    initialize();
  }

  T get_Pfa() { return Pfa * PfaRatio; }

  void assemble_C_BMB() {
    using namespace std;
    dim_t k = from_idx.size();

    // Assemble (half of) C+BMB buffer.
    switch (uplo) {
    case BLIS_UPPER:
      for (dim_t j = 0; j < k; ++j)
        for (dim_t i = 0; i < j; ++i)
          Cp(i, j) = W(i, j) + UMU(i, j);
      for (dim_t j = 0; j < k; ++j) {
        for (dim_t i = 0; i < k; ++i)
          Cp(i, j + k) = +UMV(i, j);

        Cp(j, j + k) -= T(1.0);
      }
      for (dim_t j = 0; j < k; ++j)
        for (dim_t i = 0; i < j; ++i)
          Cp(i + k, j + k) = VMV(i, j);
      break;

    default:
      cerr << "updated_tdi<T>::assemble_C_BMB:"
           << " Only upper-triangular storage is supported." << endl;
    }
  }

  // Update osi <- os[msj].
  // i.e. c+_i c_{x_j}.
  void push_update(dim_t osi, dim_t msj, bool compute_pfa) {
    using namespace std;

    // This is the k-th hopping.
    dim_t n = nelec;
    dim_t k = from_idx.size();
    dim_t osj = elem_cfg.at(msj);

    // TODO: Check bounds, duplicates, etc.
    from_idx.push_back(msj);
    to_site.push_back(osi);

    // TODO: Check for hopping-backs. 
    // This can only be handled by cancellation.
    // Singularity will emergy otherwise.

    for (dim_t i = 0; i < n; ++i) {
      U(i, k) = Xij(elem_cfg.at(i), osi) - Xij(elem_cfg.at(i), osj);
      P(i, k) = M(i, msj);
    }
    // Updated the already logged U.
    for (dim_t l = 0; l < k; ++l) {
      dim_t msl = from_idx.at(l);
      T U_jl_ = U(msj, l); ///< Backup this value before change.

      // Write updates.
      U(msl, k) = Xij(to_site.at(l), osi) - Xij(elem_cfg.at(msl), osj);
      U(msj, l) = Xij(osi, to_site.at(l)) - Xij(osj, elem_cfg.at(msl));

      // Change of Q[:, l] induced by change of U[msj, l];
      axpy(n, U(msj, l) - U_jl_, &M(0, msj), 1, &Q(0, l), 1);

      // Change of UMV[l, k] induced by change of U[msj, l].
      for (dim_t o = 0; o < k; ++o)
        UMV(l, o) += (U(msj, l) - U_jl_) * P(msj, o);
    }
    gemv(BLIS_NO_TRANSPOSE, n, n, T(1.0), &M(0, 0), M.ld, &U(0, k), 1, T(0.0),
         &Q(0, k), 1);

    switch (uplo) {
    case BLIS_UPPER:
      for (dim_t l = 0; l < k; ++l)
        W(l, k) = -Xij(to_site.at(l), osi) + Xij(elem_cfg.at(from_idx.at(l)), osj);

      // UMU needs to be recalculated.
      // TODO: Find some way to avoid recalculating UQ?
      for (dim_t o = 0; o < k + 1; ++o)
        for (dim_t l = 0; l < o; ++l)
          UMU(l, o) = dot(n, &U(0, l), 1, &Q(0, o), 1);
      for (dim_t l = 0; l < k; ++l) {
        UMV(l, k) = dot(n, &U(0, l), 1, &P(0, k), 1);
        UMV(k, l) = dot(n, &U(0, k), 1, &P(0, l), 1);
      }
      UMV(k, k) = dot(n, &U(0, k), 1, &P(0, k), 1);
      for (dim_t l = 0; l < k; ++l)
        VMV(l, k) /* -VMV(k, l) */ = -P(msj, l); ///< P(from_idx.at(l), k);
      break;

    default:
      cerr << "updated_tdi<T>::push_update:"
           << " Only upper-triangular storage is supported." << endl;
    }

    if (compute_pfa) {
      // NOTE: Update k to be new size.
      k += 1;
      // Allocate scratchpad.
      dim_t lwork = 2 * k * npanel_sub;
      T *pfwork = new T[lwork];

      // Assemble (half of) C+BMB buffer.
      assemble_C_BMB();

      // If it's the first update Pfafian can be directly read out.
      if (k == 0) {
        PfaRatio = -UMV(0, 0);
        return;
      }

      // Compute pfaffian.
      signed info = skpfa(uplo, 2 * k, &Cp(0, 0), Cp.ld, &Gc(0, 0), Gc.ld, cPov,
                          false, &PfaRatio, pfwork, lwork);
#ifdef _DEBUG
      cout << "SKPFA: info=" << info << endl;
#endif
      // Pfaffian of C = [ W -I; I 0 ].
      PfaRatio *= pow(-1.0, k * (k + 1) / 2);

      delete[] pfwork;
    } else
      // Set to 0.0 to denote dirty.
      PfaRatio = 0.0;
  }

  void push_update(dim_t osi, dim_t msj) { push_update(osi, msj, true); }

  void pop_update() {
    from_idx.pop_back();
    to_site.pop_back();

    // New update size.
    dim_t k = from_idx.size();

    // Reassemble C and scratchpads.
    assemble_C_BMB();
    dim_t lwork = 2 * k * npanel_sub;
    T *pfwork = new T[lwork];

    // Compute new (previous, in fact) Pfaffian.
    signed info = skpfa(uplo, 2 * k, &Cp(0, 0), Cp.ld, &Gc(0, 0), Gc.ld, cPov,
                        false, &PfaRatio, pfwork, lwork);
    PfaRatio *= pow(-1.0, k * (k + 1) / 2);

    delete[] pfwork;
  }

  void merge_updates() {
    using namespace std;

    dim_t n = nelec;
    dim_t k = from_idx.size();
    if (k == 0)
      return;

    // Allocate scratchpad.
    dim_t lwork = 2 * k * npanel_sub;
    T *pfwork = new T[lwork];

    if (k == 1) {
      // Trivial inverse.
      Cp(0, 1) = -1.0 / Cp(0, 1);
      Cp(1, 0) = -1.0 / Cp(1, 0);
    } else
      signed info = sktdi(uplo, 2 * k, &Cp(0, 0), Cp.ld, &Gc(0, 0), Gc.ld, cPov,
                          pfwork, lwork);
    inv_update(k, Cp);

    // Apply hopping.
    for (int j = 0; j < k; ++j)
      elem_cfg.at(from_idx.at(j)) = to_site.at(j);
    from_idx.clear();
    to_site.clear();
    Pfa *= PfaRatio;
    PfaRatio = 1.0;

    delete[] pfwork;
  }

  void inv_update(dim_t k, colmaj<T> &C) {
    colmaj<T> ABC(&U(0, 0), U.ld); ///< Use U as inv(A)*B*upper(C) buffer.
    colmaj<T> AB(&Q(0, 0), Q.ld);

    if (k == 1 && single_hop_alpha) {
      // k == 1 requires no copying
      skr2k(uplo, BLIS_NO_TRANSPOSE, nelec, 1, C(0, 1), &Q(0, 0), Q.ld,
            &P(0, 0), P.ld, T(1.0), &M(0, 0), M.ld);
      // See below for reason of calling this procedule.
      update_uplo(uplo);
      return;
    }

    // Close empty space between Q and P.
    if (k != mmax)
      for (dim_t j = 0; j < k; ++j)
        memcpy(&AB(0, k + j), &P(0, j), nelec * sizeof(T));

    // Copy AB to ABC for TRMM interface.
    for (dim_t j = 0; j < 2 * k - 1; ++j)
      // inv(A)*U  [ 0 + + +
      //             0 0 + +
      //             0 0 0 +
      //             0 0 0 0 ] => AB[:, 0:2] -> ABC[:, 1:3]
      memcpy(&ABC(0, j + 1), &AB(0, j), nelec * sizeof(T));
    trmm(BLIS_RIGHT, BLIS_UPPER, BLIS_NO_TRANSPOSE, nelec, 2 * k - 1, T(1.0),
         &C(0, 1), C.ld, &ABC(0, 1), ABC.ld);

    // Update: write to M.
    skr2k(uplo, BLIS_NO_TRANSPOSE, nelec, 2 * k - 1, T(1.0), &ABC(0, 1), ABC.ld,
          &AB(0, 1), AB.ld, T(1.0), &M(0, 0), M.ld);

    // Identity update to complete antisymmetric matrix.
    // This called due to lack of skmm support at the moment.
    update_uplo(uplo);
  }

  /**
   * Complete antisymmetric matrix.
   */
  void skcomplete(uplo_t uplo_, dim_t n, colmaj<T> &A) {
    for (dim_t j = 0; j < n; ++j) {
      for (dim_t i = 0; i < j; ++i) {
        switch (uplo_) {
        case BLIS_UPPER:
          A(j, i) = -A(i, j);
          break;

        case BLIS_LOWER:
          A(i, j) = -A(j, i);
          break;

        default:
          break;
        }
      }
      A(j, j) = T(0.0);
    }
  }

  void update_uplo(uplo_t uplo_new) {
    skcomplete(uplo /* NOTE: old uplo */, nelec, M);
    if (from_idx.size()) {
      skcomplete(uplo, from_idx.size(), UMU);
      skcomplete(uplo, from_idx.size(), VMV);
      skcomplete(uplo, from_idx.size(), W);
    }

    uplo = uplo_new;
  }

}; 

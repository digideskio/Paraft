/*******************************************************************************
 *
 * Copyright (C) 2011 Paulo Joia Filho
 * University of Sao Paulo - Sao Carlos/SP, Brazil.
 * All Rights Reserved.
 *
 * This file is part of Projection Analyzer (PAn).
 *
 * PAn is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * PAn is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public 
 * License for more details.
 *
 * This code was developed by Paulo Joia Filho <pjoia@icmc.usp.br>
 * at Institute of Mathematics and Computational Sciences - ICMC
 * University of Sao Paulo, Sao Carlos/SP, Brazil (http://www.icmc.usp.br)
 *
 * Contributor(s):  Luis Gustavo Nonato <gnonato@icmc.usp.br>
 *
 * You should have received a copy of the GNU Lesser General Public License 
 * along with PAn. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#include "pan_lamp.h"

// Member variables
static decimal m_elapsedtime = 0.0;

decimal lamp_elapsedtime()
{
    return m_elapsedtime;
}

int lamp_execute(decimal *inputdata, decimal *inputsampdata, decimal *inputsampproj,
                 struct lamp_struct *inputinfo, decimal* outputproj)
{
    if (inputinfo->projdim == DEFAULT_VALUE) inputinfo->projdim = 2;

    // dimensions
    int m = inputinfo->numpoints;
    int d = inputinfo->highdim;
    int k = inputinfo->numsamples;
    int r = inputinfo->projdim;

    clock_t start, end;
    start = clock()/(CLOCKS_PER_SEC/1000);

    // temporarily here !!!
    decimal *mySampleData = inputsampdata;
    decimal *mySampleProj = inputsampproj;

    // scalar
    decimal Wsum, Wsqrt;

    // vector
    decimal* W    = (decimal*) malloc (k*sizeof(decimal));
    decimal* X    = (decimal*) malloc (d*sizeof(decimal));
    decimal* P    = (decimal*) malloc (d*sizeof(decimal));
    decimal* Psum = (decimal*) malloc (d*sizeof(decimal));
    decimal* Pstar= (decimal*) malloc (d*sizeof(decimal));
    decimal* Phat = (decimal*) malloc (d*sizeof(decimal));
    decimal* Q    = (decimal*) malloc (r*sizeof(decimal));
    decimal* Qsum = (decimal*) malloc (r*sizeof(decimal));
    decimal* Qstar= (decimal*) malloc (r*sizeof(decimal));
    decimal* Qhat = (decimal*) malloc (r*sizeof(decimal));
    decimal* Y    = (decimal*) malloc (r*sizeof(decimal));

    // matrix
    decimal *A    = (decimal*) malloc (k*d*sizeof(decimal));
    decimal *B    = (decimal*) malloc (k*r*sizeof(decimal));
    decimal *AtB  = (decimal*) malloc (d*r*sizeof(decimal));
    decimal *M    = (decimal*) malloc (d*r*sizeof(decimal));

    // gsl object
    #if DBL_PRECISION
        gsl_matrix_view U;
        gsl_matrix *V = gsl_matrix_alloc(r, r);
        gsl_vector *S = gsl_vector_alloc(r);
    #else
        gsl_matrix_float_view U;
        gsl_matrix_float *V = gsl_matrix_float_alloc(r, r);
        gsl_vector_float *S = gsl_vector_float_alloc(r);
    #endif

    // Starting calcs
    int register p, i;
    unsigned char jump;
    for (p = 0; p < m; p++) {

        // point to be projected
        decptrcpy(X, NULL_INT, NULL_INT, inputdata, p*d, p*d+d-1);

        // selection of control points (minimal dissimilarity)
        //if (percCtrlpoints < 100.0) {
        //    selectCtrlpointsByMinimalDiss(X);
        //}

        //==============================================================
        // STEP 1: Obtain W, Pstar and Qstar
        //==============================================================
        decptrclr(Psum, d);
        decptrclr(Qsum, r);
        Wsum = 0;
        jump = 0;

        for (i = 0; i < k; i++) {
            decptrcpy(P, NULL_INT, NULL_INT, mySampleData, i*d, i*d+d-1);
            decptrcpy(Q, NULL_INT, NULL_INT, mySampleProj, i*r, i*r+r-1);

            *(W+i) = euclid2_norm(X, P, d);

            // coincident points
            if (*(W+i) == 0.0) {
                decptrcpy(outputproj, p*r, p*r+r-1, Q, NULL_INT, NULL_INT);
                jump = 1;
                break;
            }

            *(W+i) = 1 / *(W+i);

            vector_multscalar(P, *(W+i), P, d);
            vectors_add(Psum, Psum, P, d);

            vector_multscalar(Q, *(W+i), Q, r);
            vectors_add(Qsum, Qsum, Q, r);

            Wsum += *(W+i);
        }

        if (jump) continue;

        vector_multscalar(Pstar, 1/Wsum, Psum, d);
        vector_multscalar(Qstar, 1/Wsum, Qsum, r);

        //==============================================================
        // STEP 2: Obtain Phat, Qhat, A and B
        //==============================================================

        for (i = 0; i < k; i++) {

            // Phat and Qhat
            decptrcpy(P, NULL_INT, NULL_INT, mySampleData, i*d, i*d+d-1);
            vectors_subtract(Phat, P, Pstar, d);

            decptrcpy(Q, NULL_INT, NULL_INT, mySampleProj, i*r, i*r+r-1);
            vectors_subtract(Qhat, Q, Qstar, r);

            // A and B matrices
            Wsqrt = sqrt(*(W+i));

            vector_multscalar(Phat, Wsqrt, Phat, d);
            decptrcpy(A, i*d, i*d+d-1, Phat, NULL_INT, NULL_INT);

            vector_multscalar(Qhat, Wsqrt, Qhat, r);
            decptrcpy(B, i*r, i*r+r-1, Qhat, NULL_INT, NULL_INT);
        }

        //==============================================================
        // STEP 3: Projection
        //==============================================================

        // Compute AtB = At*B
        #if DBL_PRECISION
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, d, r, k,
                        1.0, A, d, B, r, 0.0, AtB, r);
        #else
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, d, r, k,
                        1.0, A, d, B, r, 0.0, AtB, r);
        #endif

        // SVD Computation
        #if DBL_PRECISION
            U = gsl_matrix_view_array(AtB, d, r);
            gsl_linalg_SV_decomp_jacobi(&U.matrix, V, S);
        #else
            U = gsl_matrix_float_view_array(AtB, d, r);
            gsl_linalg_float_SV_decomp_jacobi(&U.matrix, V, S);
        #endif

        // Point projection
        #if DBL_PRECISION
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, d, r, r,
                1.0, gsl_matrix_ptr(&U.matrix, 0, 0), r, gsl_matrix_ptr(V, 0, 0), r, 0.0, M, r);
        #else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, d, r, r,
                1.0, gsl_matrix_float_ptr(&U.matrix, 0, 0), r, gsl_matrix_float_ptr(V, 0, 0), r, 0.0, M, r);
        #endif
        vectors_subtract(X, X, Pstar, d);

        #if DBL_PRECISION
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, r, d,
                1.0, X, d, M, r, 0.0, Y, r);
        #else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, r, d,
                1.0, X, d, M, r, 0.0, Y, r);
        #endif
        vectors_add(Y, Y, Qstar, r);

        // Add point to projection
        decptrcpy(outputproj, p*r, p*r+r-1, Y, NULL_INT, NULL_INT);

        #ifdef DEBUG_LAMP
            decptr_write("out/A.out", A, ",", k, d, 0, NULL);
            decptr_write("out/B.out", B, ",", k, r, 0, NULL);
            decptr_write("out/AtB.out", AtB, ",", d, r, 0, NULL);
            #if DBL_PRECISION
                decptr_write("out/U.out", gsl_matrix_ptr(&U.matrix, 0, 0), ",", d, r, 0, NULL);
                decptr_write("out/V.out", gsl_matrix_ptr(V, 0, 0), ",", r, r, 0, NULL);
                decptr_write("out/S.out", gsl_vector_ptr(S, 0), ",", r, 1, 0, NULL);
            #else
                decptr_write("out/U.out", gsl_matrix_float_ptr(&U.matrix, 0, 0), ",", d, r, 0, NULL);
                decptr_write("out/V.out", gsl_matrix_float_ptr(V, 0, 0), ",", r, r, 0, NULL);
                decptr_write("out/S.out", gsl_vector_float_ptr(S, 0), ",", r, 1, 0, NULL);
            #endif
            decptr_write("out/M.out", M, ",", d, r, 0, NULL);
            decptr_write("out/Pstar.out", Pstar, ",", 1, d, 0, NULL);
            decptr_write("out/XminusPstar.out", X, ",", 1, d, 0, NULL);
            decptr_write("out/Qstar.out", Qstar, ",", 1, r, 0, NULL);
            decptr_write("out/YplusQstar.out", Y, ",", 1, r, 0, NULL);
        #endif
    }

    // Release memory
    free(W); free(X); free(Y);
    free(P); free(Psum); free(Pstar); free(Phat);
    free(Q); free(Qsum); free(Qstar); free(Qhat);
    free(A); free(B); free(AtB);
    free(M);
    #if DBL_PRECISION
        gsl_matrix_free(V); gsl_vector_free(S);
    #else
        gsl_matrix_float_free(V); gsl_vector_float_free(S);
    #endif

    end = clock()/(CLOCKS_PER_SEC/1000);
    m_elapsedtime = (end-start)/1000.0;

    return 0;
}

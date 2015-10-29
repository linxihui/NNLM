//[[Rcpp::depends(RcppArmadillo)]]
#include <bits/stdc++.h>
#include <RcppArmadillo.h>

using namespace std;
using namespace arma;

mat nnls_solver(mat H, mat mu, int max_iter, double tol)
Rcpp::List nmf_nnls(mat A, int k, double eta, double beta, int max_iter, double tol)

Rcpp::List nmf_ols(mat A, int k, double eta, double beta, int max_iter, double tol)

Rcpp::List nmf_brunet(mat V, int k, int max_iter, double tol)
mat get_H_brunet(mat V, mat W, int max_iter, double tol)


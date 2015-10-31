#include "nnlm.h"

//[[Rcpp::export]]
RcppExport SEXP nmf_brunet(SEXP V_, SEXP k_ , SEXP max_iter_ , SEXP tol_)
{
	/* 
	 * Description: 
	 * 	An implment of Brunet's multiplicative updates based on KL divergence for non-negative matrix factorization.
	 * Arguments:
	 * 	V: a matrix to be decomposed, such that V ~ W*H
	 * 	k: rank
	 * Return:
	 * 	A list of W, H, error, steps (= iteration until convergent or max_iter reached)
	 * Complexity:
	 * 	O(max_iter x k x V.n_row x V.n_col)
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-10-28
	 */

	mat V = Rcpp::as<mat>(V_);
	int k = Rcpp::as<int>(k_);
	int max_iter = Rcpp::as<int>(max_iter_);
	double tol = Rcpp::as<double>(tol_);

	mat W = randu(V.n_rows, k), H = randu(k, V.n_cols);
	mat Vbar = W*H; // current W*H
	rowvec w, ha; // W/h  = col/row sum of W/H
	colvec h, wa; // ha/wa = previous H.row/W.col for fast update of Vbar
	vec err(max_iter);
	err.fill(-1);

	int i = 0;
	for(; i < max_iter; i++) 
	{
		w = sum(W);
		h = sum(H, 1);
		for (int a = 0; a < k; a++)
		{
			wa = W.col(a);
			W.col(a) %= (V / Vbar) * H.row(a).t() / h.at(a);
			Vbar += (W.col(a) - wa) * H.row(a);

			ha = H.row(a);
			H.row(a) %= W.col(a).t() * (V / Vbar) / w.at(a);
			Vbar += W.col(a) * (H.row(a) - ha);
		}
		err.at(i) =  std::sqrt(mean(mean(square(V - Vbar), 1)));
		if (i > 0 && std::abs(err.at(i) - err.at(i-1)) < tol)
		{
			err.resize(i);
			break;
		}
	}

	if (max_iter <= i)
	{
		Rcpp::Function warning("warning");
		warning("Algorithm does not converge.");
	}

	return Rcpp::wrap(Rcpp::List::create(
		Rcpp::Named("W") = W,
		Rcpp::Named("H") = H,
		Rcpp::Named("error") = err
		));
}


//[[Rcpp::export]]
RcppExport SEXP get_H_brunet(SEXP V_, SEXP W_, SEXP max_iter_, SEXP tol_)
{
	/*
	 * Description:
	 * 	Get coefficient matrix H given V and W, where V ~ W*H
	 */

	mat V = Rcpp::as<mat>(V_);
	mat W = Rcpp::as<mat>(W_);
	int max_iter = Rcpp::as<int>(max_iter_);
	double tol = Rcpp::as<double>(tol_);

	int k = W.n_cols;
	mat H = randu(k, V.n_cols);
	mat Vbar = W*H;
	rowvec w = sum(W);
	colvec h, ha;
	vec err(max_iter);
	err.fill(-1);

	int i = 0;
	for(; i < max_iter; i++)
	{
		h = sum(H, 1);
		for (int a = 0; a < k; a++)
		{
			ha = H.row(a);
			H.row(a) %= W.col(a).t() * (V / Vbar) / w.at(a);
			Vbar += W.col(a) * (H.row(a) - ha);
		}
		err.at(i) =  std::sqrt(mean(mean(square(V - Vbar), 1)));
		if (i > 0 && std::abs(err.at(i) - err.at(i-1)) < tol)
		{
			err.resize(i);
			break;
		}
	}

	if (max_iter <= i)
	{
		Rcpp::Function warning("warning");
		warning("Algorithm does not converge.");
	}

	return Rcpp::wrap(H);
}


/*
/[[Rcpp::export]]
mat get_W_brunet(mat V, mat H, int max_iter = 500, double tol = 1e-5)
{
	// Description: Get profile matrix W given V and H, where V ~ W*H

	int k = H.n_rows;
	mat W = randu(V.n_rows, k);
	mat Vbar = W*H;
	colvec h = sum(h, 1);
	rowvec w, wa;
	vec err(max_iter);
	err.fill(-1);

	int i = 0;
	for(; i < max_iter; i++)
	{
		w = sum(W);
		for (int a = 0; a < k; a++)
		{
			wa = W.col(a);
			W.col(a) %= (V / Vbar) * H.row(a).t() / h.at(a);
			Vbar += (W.col(a) - wa) * H.row(a);
		}
		err.at(i) =  std::sqrt(mean(mean(square(V - Vbar), 1)));
		if (i > 0 && std::abs(err.at(i) - err.at(i-1)) < tol)
		{
			err.resize(i+1);
			break;
		}
	}

	if (max_iter <= i)
	{
		Rcpp::Function warning("warning");
		warning("Algorithm does not converge.");
	}

	return W;
}
*/

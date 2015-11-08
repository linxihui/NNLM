#include "nnlm.h"


//[[Rcpp::export]]
Rcpp::List nmf_brunet(const mat & A, int k, int max_iter , double tol, int n_threads, bool show_progress, bool show_warning)
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
	 * 	2015-10-31
	 */


	mat W = randu(A.n_rows, k), H = randu(k, A.n_cols);
	mat Abar = W*H; // current W*H
	rowvec w, ha; // W/h  = col/row sum of W/H
	colvec h, wa; // ha/wa = previous H.row/W.col for fast update of Abar
	vec err(max_iter), trgt_err(max_iter);
	trgt_err.fill(-1);

	// check progression
	Progress prgrss(max_iter, show_progress);

	int i = 0;
	for(; i < max_iter; i++) 
	{
		Rcpp::checkUserInterrupt();

		prgrss.increment();

		w = sum(W);
		h = sum(H, 1);
		for (int a = 0; a < k; a++)
		{
			wa = W.col(a);
			W.col(a) %= (A / Abar) * H.row(a).t() / h.at(a);
			Abar += (W.col(a) - wa) * H.row(a);

			ha = H.row(a);
			H.row(a) %= W.col(a).t() * (A / Abar) / w.at(a);
			Abar += W.col(a) * (H.row(a) - ha);
		}
		err.at(i) =  std::sqrt(mean(mean(square(A - Abar))));
		trgt_err.at(i) = accu(A % (arma::trunc_log(A) - arma::trunc_log(Abar)) - A - Abar);
		if (i > 0 && std::abs(trgt_err.at(i) - trgt_err.at(i-1))/(std::abs(trgt_err.at(i-1)) + 1e-6) < tol)
			break;
	}

	if (show_warning && max_iter <= i)
		Rcpp::warning("Target tolerence not reached. Try a larger max.iter.");

	err.resize(i < max_iter ? i+1 : max_iter);
	trgt_err.resize(err.n_elem);

	return Rcpp::List::create(
		Rcpp::Named("W") = W,
		Rcpp::Named("H") = H,
		Rcpp::Named("error") = err,
		Rcpp::Named("target_error") = trgt_err
		);
}

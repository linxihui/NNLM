#include "nnlm.h"


//[[Rcpp::export]]
mat get_H_brunet(const mat & A, const mat & W,  int max_iter, double tol, int n_threads, bool show_progress)
{
	/*
	 * Description:
	 * 	Get coefficient matrix H given A and W, where A ~ W*H
	 */

	int k = W.n_cols;
	mat H = randu(k, A.n_cols);
	mat Abar = W*H;
	rowvec w = sum(W), ha;
	colvec h;
	vec err(max_iter);
	err.fill(-1);

	// check progression
	Progress prgrss(max_iter, show_progress);

	int i = 0;
	for(; i < max_iter; i++)
	{
		Rcpp::checkUserInterrupt();

		prgrss.increment();

		h = sum(H, 1);
		for (int a = 0; a < k; a++)
		{
			ha = H.row(a);
			H.row(a) %= W.col(a).t() * (A / Abar) / w.at(a);
			Abar += W.col(a) * (H.row(a) - ha);
		}
		err.at(i) = accu(A % (arma::trunc_log(A) - arma::trunc_log(Abar)) - A - Abar);
		if (i > 0 && std::abs(err.at(i) - err.at(i-1))/(std::abs(err.at(i-1)) + 1e-6) < tol)
			break;
	}


	if (max_iter <= i)
	{
		err.resize(max_iter);
		Rcpp::warning("Target tolerence not reached. Try a larger max.iter.");
	}
	else
	{
		err.resize(i+1);
	}

	return H;
}

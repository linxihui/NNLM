#include <nnlm.hpp>

//[[Rcpp::export]]
Rcpp::List nmf_brunet(mat V, int k = 1, int max_iter = 500, double tol = 1e-5)
{
	/* 
	 * Description: 
	 * 	An implment of Brunet's multiplicative updates based on KL divergence for non-negative matrix factorization.
	 * 	This is more stable than the following one as it alternatingly updates each rank of W and H
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
			ha = H.row(a);
			H.row(a) %= W.col(a).t() * (V / Vbar) / w.at(a);
			Vbar += W.col(a) * (H.row(a) - ha);
			wa = W.col(a);
			W.col(a) %= (V / Vbar) * H.row(a).t() / h.at(a);
			Vbar += (W.col(a) - wa) * H.row(a);
		}
		err.at(i) =  sqrt(mean(mean(square(V - Vbar), 1)));
		if (i > 0 && abs(err.at(i) - err.at(i-1)) < tol)
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

	return Rcpp::List::create(
		Rcpp::Named("W") = W,
		Rcpp::Named("H") = H,
		Rcpp::Named("error") = err
		);
}


//[[Rcpp::export]]
mat get_H_brunet(mat V, mat W, int max_iter = 500, double tol = 1e-5)
{
	/*
	 * Description:
	 * 	Get coefficient matrix H given V and W, where V ~ W*H
	 */

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
		err.at(i) =  sqrt(mean(mean(square(V - Vbar), 1)));
		if (i > 0 && abs(err.at(i) - err.at(i-1)) < tol)
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

	return H;
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
		err.at(i) =  sqrt(mean(mean(square(V - Vbar), 1)));
		if (i > 0 && abs(err.at(i) - err.at(i-1)) < tol)
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

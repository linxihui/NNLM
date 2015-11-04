#include "nnlm.h"


//[[Rcpp::export]]
Rcpp::List nnls(const mat & A, const mat & b, int max_iter, double tol, int n_threads, bool show_progress)
{
	/*
	 * Description: sequential Coordinate-wise algorithm for non-negative least square regression problem
	 * 		A x = b, s.t. x >= 0
	 * Arguments:
	 * 	A, b: see above
	 * 	max_iter: maximum number of iterations.
	 * 	tol: stop criterion, relative change on x between two successive iteration.
	 * Return: 
	 * 	x: solution to argmin_{x, x>=0} ||Ax - b||_F^2
	 * Reference: 
	 * 	http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-TR-2005-06.pdf 
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-10-31
	 */

	// This following code is duplicated with nnls_solve in nmf_nnls.cpp, as there is an InterruptableProgressMonitor conflit when use Progress::check_abort

	mat H = A.t()*A;
	mat mu = -A.t()*b;

	mat x(H.n_cols, mu.n_cols, fill::zeros);
	if (n_threads < 0) n_threads = 0; 

	vec err(mu.n_cols);
	vec rel_err(mu.n_cols);
	vec iter(mu.n_cols);

	err.fill(-1);
	rel_err.fill(-1);
	iter.fill(-1);

	Progress p((mu.n_cols*max_iter)/10 + 1, show_progress);

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
	for (int j = 0; j < mu.n_cols; j++)
	{
		vec x0(H.n_cols);
		x0.fill(-9999);
		double tmp;
		int i = 0;
		double err1; //last error
		double err2 = 9999; //current error
		do {
			if (i % 10 == 0)
			{
				if (Progress::check_abort())
					break;
				p.increment(); // update progress
			}
			x0 = x.col(j);
			for (int k = 0; k < H.n_cols; k++)
			{
				tmp = x.at(k,j) - mu.at(k,j) / H.at(k,k);
				if (tmp < 0) tmp = 0;
				if (tmp != x.at(k,j)) mu.col(j) += (tmp - x.at(k, j)) * H.col(k);
				x.at(k,j) = tmp;
			}
			err1 = err2;
			err2 = arma::max(arma::abs(x.col(j) - x0));
		} while(++i < max_iter && std::abs(err1 - err2) / (err1 + 1e-6) > tol);

		err[j] = err2;
		rel_err[j] = std::abs(err1 - err2) / (err1 + 1e-6);
		iter[j] = i;
	}

	return Rcpp::List::create(
		Rcpp::Named("x") = x,
		Rcpp::Named("iteration") = iter,
		Rcpp::Named("abs.err") = err,
		Rcpp::Named("rel.err") = rel_err
		);
}

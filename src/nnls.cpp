#include "nnlm.h"


//[[Rcpp::export]]
mat nnls(const mat & A, const mat & b, int max_iter, double tol, int n_threads, bool show_progress)
{
	/*
	 * Description: sequential Coordinate-wise algorithm for non-negative least square regression problem
	 * 		A x = b, s.t. x >= 0
	 * Arguments:
	 * 	A, b: see above
	 * 	max_iter: maximum number of iterations.
	 * 	tol: stop criterion, minimum change on x between two successive iteration.
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
	Progress p(mu.n_cols*max_iter, show_progress);
	if (n_threads < 0) n_threads = 0; 

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
	for (int j = 0; j < mu.n_cols; j++)
	{
		vec x0(H.n_cols);
		x0.fill(-9999);
		double tmp;
		int i = 0;
		while(i < max_iter && arma::max(arma::abs(x.col(j) - x0)) > tol)
		{
			if (!Progress::check_abort()) 
			{
				p.increment(); // update progress
				x0 = x.col(j);
				for (int k = 0; k < H.n_cols; k++) 
				{
					tmp = x.at(k,j) - mu.at(k,j) / H.at(k,k);
					if (tmp < 0) tmp = 0;
					if (tmp != x.at(k,j)) mu.col(j) += (tmp - x.at(k, j)) * H.col(k);
					x.at(k,j) = tmp;
				}
			}
			++i;
		}
	}

	return x;
}

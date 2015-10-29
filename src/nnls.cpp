#include "nnlm.hpp"


//[[Rcpp::export]]
RcppExport mat nnls(mat A, mat b, int max_iter = 500, double tol = 1e-6)
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
	 * 	2015-10-28
	 */

	if(A.n_rows != b.n_rows)
		throw std::invalid_argument("A and b must have the same number of rows.");
	
	return nnls_solver(A.t()*A, -A.t()*b, max_iter, tol);
}


mat nnls_solver(mat H, mat mu, int max_iter = 500, double tol = 1e-6)
{
	/*
	 * Description: sequential Coordinate-wise algorithm for non-negative least square regression problem
	 * 		A x = b, s.t. x >= 0
	 * Arguments:
	 * 	H: A^T * A
	 * 	mu: -A^T * b
	 * 	max_iter: maximum number of iterations.
	 * 	tol: stop criterion, minimum change on x between two successive iteration.
	 * Return: 
	 * 	x: solution to argmin_{x, x>=0} ||Ax - b||_F^2
	 * Reference: 
	 * 	http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-TR-2005-06.pdf 
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-10-28
	 */

	mat x(H.n_cols, mu.n_cols, fill::zeros);
	vec x0(H.n_cols);
	double tmp;

	for (int j = 0; j < mu.n_cols; j++)
	{
		x0.fill(-9999);
		int i = 0;
		while(i < max_iter && max(abs(x.col(j) - x0)) > tol) 
		{
			x0 = x.col(j);
			for (int k = 0; k < H.n_cols; k++) 
			{
				tmp = x.at(k,j) - mu.at(k,j) / H.at(k,k);
				if (tmp < 0) tmp = 0;
				if (tmp != x.at(k,j)) mu.col(j) += (tmp - x.at(k, j)) * H.col(k);
				x.at(k,j) = tmp;
			}
			++i;
		}
	}

	return x;
}

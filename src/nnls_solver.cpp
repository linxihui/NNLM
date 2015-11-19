#include "nnlm.h"


double mse(const mat & A, const mat & W, const mat & H, const mat & W1, const mat & H2)
{
	// compute mean square error of A and fixed A
	const int k = W.n_cols - H2.n_cols;

	mat Adiff = A;
	Adiff -= W.cols(0, k-1) * H.cols(0, k-1).t();
	if (!W1.empty())
		Adiff -= W1*H.cols(k, H.n_cols-1).t();
	if (!H2.empty())
		Adiff -= W.cols(k, W.n_cols-1)*H2.t();

	if (A.is_finite())
		return mean(mean(square(Adiff)));
	else
		return mean(square(Adiff.elem(find_finite(Adiff))));
}

inline void update_WtW(mat & WtW, const mat & W, const mat & W1)
{
	// compute WtW = (W, W1)^T (W, W1)
	int k = W.n_cols;
	if (W1.empty())
		WtW = W.t()*W;
	else
	{
		int kk = k + W1.n_cols;
		WtW.submat(0, 0, k-1, k-1) = W.t() * W;
		WtW.submat(k, 0, kk-1, k-1) = W1.t() * W;
		WtW.submat(0, k, k-1, kk-1) = WtW.submat(k, 0, kk-1, k-1).t();
		WtW.submat(k, k, kk-1, kk-1) = W1.t() * W1;
	}
}

inline void update_WtW(mat & WtW, const mat & W, const mat & W1, const mat & H2)
{
	// compute WtW = (W[:, 0:k-1], W1)^T (W[:, 0:k-1], W1)
	if (H2.empty())
		update_WtW(WtW, W, W1);
	else
	{
		int k = W.n_cols - H2.n_cols;
		update_WtW(WtW, W.cols(0, k-1), W1);
	}
}

inline void update_WtA(mat & WtA, const mat & W, const mat & W1, const mat & A)
{
	// compute WtA  = (W, W1)^T A
	int k = W.n_cols;
	if (W1.empty())
	{
		//std::cout << "1.32" << std::endl;
		WtA = -W.t()*A;
	}
	else
	{
		//std::cout << "1.4" << std::endl;
		WtA.rows(0, k-1) = -W.t()*A;
		WtA.rows(k, WtA.n_rows-1) = -W1.t()*A;
	}
}

inline void update_WtA(mat & WtA, const mat & W, const mat & W1, const mat & H2, const mat & A)
{
	// compute WtA = (W[:, 0:k-1], W1)^T (A - W[, k:end] H2^T)
	if (H2.empty())
		update_WtA(WtA, W, W1, A);
	else
	{
		int k = W.n_cols - H2.n_cols;
		//std::cout << "1.3" << std::endl;
		//A.print("A = ");
		//(W.cols(k, W.n_cols-1) * H2.t()).print("W[, k:] = ");
		update_WtA(WtA, W.cols(0, k-1), W1, A - W.cols(k, W.n_cols-1) * H2.t());
	}
}

mat nnls_solver(const mat & H, mat mu, const umat & mask, int max_iter, double rel_tol, int n_threads)
{
	/****************************************************************************************************
	 * Description: sequential Coordinate-wise algorithm for non-negative least square regression problem
	 * 		A x = b, s.t. x[!m] >= 0, x[m] == 0
	 * Arguments:
	 * 	H         : A^T * A
	 * 	mu        : -A^T * b
	 * 	mask      : a mask matrix (m) of same dim of x
	 * 	max_iter  : maximum number of iterations
	 * 	rel_tol   : stop criterion, minimum change on x between two successive iteration
	 * 	n_threads : number of threads
	 * Return:
	 * 	x : solution to argmin_{x, x>=0} ||Ax - b||_F^2
	 * Reference:
	 * 	http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-TR-2005-06.pdf
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-11-16
	 ****************************************************************************************************/

	mat x(H.n_cols, mu.n_cols, fill::zeros);
	if (n_threads < 0) n_threads = 0;
	bool is_masked = !mask.empty();


	#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
	for (int j = 0; j < mu.n_cols; j++)
	{
		if (is_masked && arma::all(mask.col(j))) 
			continue;
		vec x0(H.n_cols);
		// x0.fill(-9999);
		double tmp;
		int i = 0;
		double err1, err2 = 9999;
		do {
			// break if all entries of col_j are masked
			x0 = x.col(j);
			err1 = err2;
			err2 = 0;
			for (int k = 0; k < H.n_cols; k++)
			{
				if (is_masked && mask(k,j) > 0) continue;
				tmp = x(k,j) - mu(k,j) / H(k,k);
				if (tmp < 0) tmp = 0;
				if (tmp != x(k,j))
				{
					mu.col(j) += (tmp - x(k,j)) * H.col(k);
				}
				x(k,j) = tmp;
				tmp = std::abs(x(k,j) - x0(k));
				if (tmp > err2) err2 = tmp;
			}
		} while(++i < max_iter && std::abs(err1 - err2) / (err1 + 1e-9) > rel_tol);
	}
	return x;
}


mat nnls_solver_without_missing(mat & WtW, mat & WtA,
	const mat & A, const mat & W, const mat & W1, const mat & H2, const umat & mask, 
	const double & eta, const double & beta, int max_iter, double rel_tol, int n_threads)
{

	// A = [W, W1, W2] [H, H1, H2]^T.
	// Where A has not missing
	// WtW and WtA are auxiliary matrices, passed by referenced and can be modified 
	update_WtW(WtW, W, W1, H2);
	update_WtA(WtA, W, W1, H2, A);
	if (beta > 0) WtW += beta;
	if (eta > 0) WtW.diag() += eta;
	
	return nnls_solver(WtW, WtA, mask, max_iter, rel_tol, n_threads);
}


mat nnls_solver_with_missing(const mat & A, const mat & W, const mat & W1, const mat & H2, const umat & mask, 
	const double & eta, const double & beta, int max_iter, double rel_tol, int n_threads)
{
	// A = [W, W1, W2] [H, H1, H2]^T.
	// Where A may have missing values
	// Note that here in the input W = [W, W2]
	// compute x = [H, H1]^T given W, W2
	// A0 = W2*H2 is empty when H2 is empty (no partial info in H)
	// Return: x = [H, H1]

	int n = A.n_rows, m = A.n_cols;
	int k = W.n_cols - H2.n_cols;
	int kW = W1.n_cols;
	int nH = k+kW;

	mat x(nH, m, fill::zeros);

	if (n_threads < 0) n_threads = 0;
	bool is_masked = !mask.empty();

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
	for (int j = 0; j < m; j++)
	{
		// break if all entries of col_j are masked
		if (is_masked && arma::all(mask.col(j))) 
			continue;
		
		uvec non_missing = find_finite(A.col(j));
		mat WtW(nH, nH); // WtW
		update_WtW(WtW, W.rows(non_missing), W1.rows(non_missing), H2);
		if (beta > 0) WtW += beta;
		if (eta > 0) WtW.diag() += eta;

		mat mu(nH, 1); // -WtA
		uvec jv(1);
		jv(0) = j;
		//non_missing.t().print("non_missing = ");
		//std::cout << "1.1" << std::endl;
		if (H2.empty())
			update_WtA(mu, W.rows(non_missing), W1.rows(non_missing), H2, A.submat(non_missing, jv));
		else
			update_WtA(mu, W.rows(non_missing), W1.rows(non_missing), H2.rows(j, j), A.submat(non_missing, jv));
		//std::cout << "1.5" << std::endl;

		vec x0(nH);
		double tmp;
		int i = 0;
		double err1, err2 = 9999;
		do {
			x0 = x.col(j);
			err1 = err2;
			err2 = 0;
			for (int l = 0; l < nH; l++)
			{
				if (is_masked && mask(l,j) > 0) continue;
				tmp = x(l,j) - mu(l,0) / WtW(l,l);
				if (tmp < 0) tmp = 0;
				if (tmp != x(l,j))
				{
					mu.col(0) += (tmp - x(l,j)) * WtW.col(l);
				}
				x(l,j) = tmp;
				tmp = std::abs(x(l,j) - x0(l));
				if (tmp > err2) err2 = tmp;
			}
		} while(++i < max_iter && std::abs(err1 - err2) / (err1 + 1e-9) > rel_tol);
	}
	return x;
}

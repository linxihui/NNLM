#include "nnlm.h"

int update(mat & H, const mat & Wt, const mat & A, const umat & mask,
	const vec & beta, int max_iter, double rel_tol, int n_threads, int method)
{
	// A = W H, solve H
	// No missing in A, Wt = W^T
	// method: 1 = scd, 2 = lee_ls, 3 = scd_kl, 4 = lee_kl

	int n = A.n_rows, m = A.n_cols;
	int K = H.n_rows;
	int total_raw_iter = 0;

	if (n_threads < 0) n_threads = 0;
	bool is_masked = !mask.empty();
	mat WtW;
	vec mu, sumW;
	if (method == 1 || method == 2)
	{
		WtW = Wt*Wt.t();
		if (beta(0) != beta(1))
			WtW.diag() += beta(0) - beta(1);
		if (beta(1) != 0)
			WtW += beta(1);
	}
	else
		sumW = sum(Wt, 1);

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic) private(mu)
	for (int j = 0; j < m; j++) // by rows of H
	{
		// break if all entries of col_j are masked
		if (is_masked && arma::all(mask.row(j)))
			continue;

		int iter;
		if (method == 1)
		{
			mu = WtW*H.col(j) - Wt*A.col(j);
			if (beta(2) != 0)
				mu += beta(2);
			iter = scd_ls_update(H.col(j), WtW, mu, mask.col(j), max_iter, rel_tol);
		}
		else if (method == 2)
			iter = lee_ls_update(H.col(j), WtW, Wt*A.col(j), beta(2), mask.col(j), max_iter, rel_tol);
		else if (method == 3)
			iter = scd_kl_update(H.col(j), Wt, A.col(j), sumW, mask.col(j), max_iter, rel_tol, beta);
		else if (method == 4)
			iter = lee_kl_update(H.col(j), Wt, A.col(j), sumW, mask.col(j), max_iter, rel_tol, beta);

		#pragma omp critical
		total_raw_iter += iter;
	}
	return total_raw_iter;
}

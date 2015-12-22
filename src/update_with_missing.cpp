#include "nnlm.h"

int update(mat & H, const mat & Wt, const mat & A, const umat & mask,
	const vec & beta, unsigned int max_iter, double rel_tol, int n_threads, int method)
{
	// A = W H, solve H
	// No missing in A, Wt = W^T
	// method: 1 = scd, 2 = lee_ls, 3 = scd_kl, 4 = lee_kl

	unsigned int m = A.n_cols;
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
		WtW.diag() += TINY_NUM; // for stability: avoid divided by 0 in scd_ls, scd_kl
	}
	else
		sumW = sum(Wt, 1);

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic) private(mu)
	for (unsigned int j = 0; j < m; j++) // by columns of H
	{
		// break if all entries of col_j are masked
		if (is_masked && arma::all(mask.col(j)))
			continue;

		int iter = 0;
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
			iter = scd_kl_update(H.col(j), Wt, A.col(j), sumW, mask.col(j), beta, max_iter, rel_tol);
		else if (method == 4)
			iter = lee_kl_update(H.col(j), Wt, A.col(j), sumW, mask.col(j), beta, max_iter, rel_tol);

		#pragma omp critical
		total_raw_iter += iter;
	}
	return total_raw_iter;
}


int update_with_missing(mat & H, const mat & Wt, const mat & A, const umat & mask,
	const vec & beta, unsigned int max_iter, double rel_tol, int n_threads, int method)
{
	// A = W H, solve H
	// With missings in A, Wt = W^T
	// method: 1 = scd, 2 = lee_ls, 3 = scd_kl, 4 = lee_kl

	unsigned int n = A.n_rows, m = A.n_cols;
	unsigned int total_raw_iter = 0;

	if (n_threads < 0) n_threads = 0;
	bool is_masked = mask.n_elem > 0;
	mat WtW;
	vec mu;

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic) private(WtW, mu)
	for (unsigned int j = 0; j < m; j++) // by columns of H
	{
		// break if all entries of col_j are masked
		if (is_masked && arma::all(mask.col(j)))
			continue;

		bool any_missing = !is_finite(A.col(j));
		uvec non_missing;
		if (any_missing)
			non_missing = find_finite(A.col(j));
	
		if (method == 1 || method == 2)
		{
			if (any_missing)
			{
				//non_missing.print("Non Missing");
				WtW = Wt.cols(non_missing)*Wt.cols(non_missing).t();
				mu = Wt.cols(non_missing) * A.elem(j*n + non_missing);
			}
			else
			{
				WtW = Wt*Wt.t();
				mu = Wt*A.col(j);
			}
			if (beta(0) != beta(1))
				WtW.diag() += beta(0) - beta(1);
			if (beta(1) != 0)
				WtW += beta(1);
		}

		int iter = 0;
		if (method == 1)
		{
			mu = WtW*H.col(j)-mu;
			if (beta(2) != 0)
				mu += beta(2);
			iter = scd_ls_update(H.col(j), WtW, mu, mask.col(j), max_iter, rel_tol);
		}
		else if (method == 2)
		{
			iter = lee_ls_update(H.col(j), WtW, mu, beta(2), mask.col(j), max_iter, rel_tol);
		}
		else if (method == 3)
		{
			if (any_missing)
				iter = scd_kl_update(H.col(j), Wt.cols(non_missing), A.elem(j*n + non_missing),
					sum(Wt.cols(non_missing), 1), mask.col(j), beta, max_iter, rel_tol);
			else
				iter = scd_kl_update(H.col(j), Wt, A.col(j), sum(Wt, 1), mask.col(j), beta, max_iter, rel_tol);
		}
		else if (method == 4)
		{
			if (any_missing)
				iter = lee_kl_update(H.col(j), Wt.cols(non_missing), A.elem(j*n + non_missing),
					sum(Wt.cols(non_missing), 1), mask.col(j), beta, max_iter, rel_tol);
			else
				iter = lee_kl_update(H.col(j), Wt, A.col(j), sum(Wt, 1), mask.col(j), beta, max_iter, rel_tol);
		}

		#pragma omp critical
		total_raw_iter += iter;
	}
	return total_raw_iter;
}

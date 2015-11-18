#include "nnlm.h"

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
		WtA = -W.t()*A;
	else
	{
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
		update_WtA(WtA, W.cols(0, k-1), W1, A - W.cols(k, W.n_cols-1) * H2.t());
	}
}

inline double mse(const mat & A, const mat & W, const mat & H, const mat & W1, const mat & H2)
{
	// compute mean square error of A and fixed A
	int k = W.n_cols - H2.n_cols;
	mat Adiff = A;
	Adiff -= W.cols(0, k-1) * H.cols(0, k-1).t();
	if (!W1.empty())
		Adiff -= W1*H.cols(k, H.n_cols-1).t();
	if (!H2.empty())
		Adiff -= W.cols(k, W.n_cols-1)*H2.t();
	return mean(mean(square(Adiff)));
}

//[[Rcpp::export]]
Rcpp::List nnmf_generalized(const mat & A, const mat & W1, const mat & H2, mat W, umat Wm, umat Hm,
	int k, double eta, double beta, int max_iter, double rel_tol, 
	int n_threads, bool show_progress, bool show_warning,
	int nnls_max_iter, double nnls_rel_tol, int trace)
{
	/******************************************************************************************************
	 *     Non-negative Matrix Factorization(NNMF) using alternating Non-negative Least Square(NNLS)
	 *     -----------------------------------------------------------------------------------------
	 * Description:
	 * 	Decompose matrix A such that
	 * 		A = [W W1 W2] [H H1 H2]^T
	 * 	where W1 and H2 are known, and [W, W2], [H, H2] could be partially fixed to 0
	 * Argument:
	 * 	A             : matrix to be decomposed
	 * 	W1, H2        : known W and H profiles. Can be emty matrices
	 * 	W             : intial of [W, W2]. Can be empty matrices
	 * 	Wm, Hm        : masks of [W, W2], [H, H1], s.t. masked entries are fixed to 0. Can be emty matrices
	 * 	k             : rank or cols of W, and k + col(H1) = col(W) = col(Wm), k + col(W1) = col(Hm)
	 * 	eta           : L2 constraint on W, W2 (non-masked entries)
	 * 	beta          : L1 constraint on H, H1 (non-masked entries)
	 * 	max_iter      : maximum number of iteration
	 * 	rel_tol       : relative tolerance between two successive iterations
	 * 	n_threads     : number of threads (openMP)
	 * 	show_progress : if to show progress, useful for long computation, suppressed if trace > 0 (on)
	 * 	trace         : # of iteractions for each print out. No trace if trace <= 0
	 * 	show_warning  : if to show warning if targeted `tol` is not reached
	 * 	nnls_max_iter : maximum iterations passed to `nnls_solver`, fixed
	 * 	nnls_rel_tol  : relative tolerance passed to `nnls_solver` for the first iteraction. 
	 * 	              : rel_tol decreases after each iteraction (divided by 1+iteraction)
	 * Return:
	 * 	A list (Rcpp::List) of W = [W, W2] and H = [H, H1]^T, iteration, 
	 * 	error and target_error (with constraints)
	 * Methods:
	 * 	Apply `nnls_solver` to [W, W2] (fixed others) and [H, H1] (fixed othes) alternatively
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-11-17
	 ******************************************************************************************************/

	int n = A.n_rows, m = A.n_cols;
	int k1 = W1.n_cols, k2 = H2.n_cols;
	int nW = k+k2, nH = k+k1;

	vec err(max_iter);
	err.fill(-9999);
	vec perr(err);

	mat WtW(nH, nH); // known [w w1], get [h h1]
	mat WtA(nH, m);
	mat HtH(nW, nW); // known [h h2], get [w w2]
	mat HtAt(nW, n);

	mat H(n, nH); //W = [w w2]
	inplace_trans(Hm);
	inplace_trans(Wm);

	if (W.empty())
	{
		W.set_size(n, nW);
		if (H2.empty()) 
		{
			W.randu();
			W = normalise(W);
		}
		else
		{
			W.cols(0, k-1).randu();
			W.cols(0, k-1) = normalise(W.cols(0, k-1));
			W.cols(k, nW-1).zeros();
		}
	}

	// check progression if tracing off
	if (trace > 0) show_progress = false;
	Progress prgrss(max_iter, show_progress);

	// solve H = [h h1] given [w w1]
	update_WtW(WtW, W, W1, H2);
	update_WtA(WtA, W, W1, H2, A);
	if (beta > 0) WtW += beta;
	H = nnls_solver(WtW, WtA, Hm, nnls_max_iter, nnls_rel_tol, n_threads).t();

	if (trace > 0)
	{
		Rprintf("%10s | %10s | %10s | %10s\n", "Iteration", "MSE", "target err", "rel err");
		Rprintf("-------------------------------------------------\n");
	}

	int i = 0;
	double rel_err = rel_tol + 1;
	double perr_last = 9999;

	double nnls_rel_tol_adp = nnls_rel_tol;

	for(; i < max_iter && std::abs(rel_err) > rel_tol; i++)
	{
		// check keyboard interrupt event
		Rcpp::checkUserInterrupt();
		prgrss.increment();

		// increasing accuracy
		nnls_rel_tol_adp = std::max(nnls_rel_tol/(i+1), 1e-8);

		// solve W = [w w2] given [h h2]
		update_WtW(HtH, H, H2, W1);
		update_WtA(HtAt, H, H2, W1, A.t());
		if (eta > 0) HtH.diag() += eta;
		W = nnls_solver(HtH, HtAt, Wm, nnls_max_iter, nnls_rel_tol_adp, n_threads).t();

		// solve H = [h h1] given [w w1]
		update_WtW(WtW, W, W1, H2);
		update_WtA(WtA, W, W1, H2, A);
		if (beta > 0) WtW += beta;
		H = nnls_solver(WtW, WtA, Hm, nnls_max_iter, nnls_rel_tol_adp, n_threads).t();

		perr(i) = mse(A, W, H, W1, H2);
		err(i) = std::sqrt(perr(i));

		if (beta > 0)
			perr(i) += mean(vectorise(square(mean(H))))*k/A.n_cols;
		if (eta > 0)
			perr(i) += eta * mean(mean(square(W)))*k*k/A.n_rows;

		perr(i) = std::sqrt(perr(i));
		rel_err = (perr_last - perr(i)) / (perr_last + 1e-6);
		if (trace > 0 && ((i+1) % trace == 0 || i == 0))
			Rprintf("%10d | %10.4f | %10.4f | %10.g\n", i+1, err(i), perr(i), rel_err);
		perr_last = perr(i);
	}

	if (i >= max_iter)
	{
		if (show_warning && std::abs(perr(i-1)-perr(i))/(perr(i-1)+1e-6) > rel_tol)
			Rcpp::warning("Target tolerance not reached. Try a larger max.iter.");
	}
	else
	{
		err.resize(i);
		perr.resize(i);
	}

	return Rcpp::List::create(
		Rcpp::Named("W") = W,
		Rcpp::Named("H") = H.t(),
		Rcpp::Named("error") = err,
		Rcpp::Named("target_error") = perr
		);
}

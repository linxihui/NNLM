#include "nnlm.h"


//[[Rcpp::export]]
Rcpp::List nmf_nnls(const mat & A, int k, double eta, double beta, int max_iter, double tol, int n_threads, bool show_progress, bool show_warning)
{
	/*
	 * Description:
	 * 	Non-negative matrix factorization with penalties using NNLS.
	 * 		argmin_{W>=0, H>=0} ||A - WH||_2^2 + eta*||W||_F^2 + beta*sum(||H.col(j)||_1^2)
	 * Arguments:
	 * 	A: matrix to be factorized as A_{n,m} ~ W_{n,k} * H_{k,m}.
	 * 	k: rank of factorization.
	 * 	eta: L2 penalty on the left (W). Default to no penalty. If eta < 0 then eta = max(A)
	 * 	beta: L1 penalty on the right (H). Default to no penalty.
	 * 	max_iter: maximum iteration of alternating NNLS solutions to H and W
	 * 	tol: stop criterion, maximum difference of target_error between two successive iterations.
	 * Return:
	 * 	A list of 
	 * 		W, H, 
	 * 		error: root mean square error between A and W*H)
	 * 		target_error: root mean(devided by nxm) square error of the target function. Same as error if no penalty.
	 * Methods:
	 * 	Apply `nnls_solver` to W and H alternatingly. `nnls_solver` is implemented using sequential coordinate descend methods.
	 * Author: 
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-10-31
	 */

	mat W(A.n_rows, k, fill::randu);
	mat H(k, A.n_cols);
	W = normalise(W);  
	mat WtW(W.n_cols, W.n_cols);
	mat HHt(H.n_rows, H.n_rows);
	// err, pen_err = root mean square error/penalized_error
	vec err(max_iter);
	err.fill(-9999);
	vec pen_err(err);

	// check progression
	Progress prgrss(max_iter, show_progress);

	// iterations and tolerance of internal nnls_solver
	int nnls_max_iter = 500;
	double nnls_tol = tol > 1e-4 ? 1e-5 : tol/10.0;

	umat mask;

	// solve H given W
	WtW = W.t()*W;
	if (beta > 0) WtW += beta;
	H = nnls_solver(WtW, -W.t()*A, mask, nnls_max_iter, nnls_tol, n_threads);

	
	prgrss.increment();

	int i = 0;

	for(; i < max_iter; i++)
	{
		Rcpp::checkUserInterrupt();

		prgrss.increment();

		// solve W given H
		HHt = H*H.t();
		if (eta > 0) HHt.diag() += eta;
		W = nnls_solver(HHt, -H*A.t(), mask, nnls_max_iter, nnls_tol, n_threads).t();

		// solve H given W
		WtW = W.t()*W;
		if (beta > 0) WtW += beta;
		H = nnls_solver(WtW, -W.t()*A, mask, nnls_max_iter, nnls_tol, n_threads);

		pen_err[i] = mean(mean(square(A - W*H)));
		err[i] = std::sqrt(pen_err[i]);
		if (beta > 0) pen_err[i] += mean(vectorise(square(mean(H)))) * k/A.n_cols;
		if (eta > 0) pen_err[i] += eta * mean(mean(square(W)))*k*k / A.n_rows;
		pen_err[i] = std::sqrt(pen_err[i]);

		if (i > 0 && std::abs(pen_err[i-1] - pen_err[i]) / (pen_err[i-1] + 1e-6) < tol) 
			break;
	}

	if (show_warning && max_iter <= i)
		Rcpp::warning("Target tolerance not reached. Try a larger max.iter.");

	err.resize(i < max_iter ? i+1 : max_iter);
	pen_err.resize(err.n_elem);

	return Rcpp::List::create(
		Rcpp::Named("W") = W, 
		Rcpp::Named("H") = H, 
		Rcpp::Named("error") = err,
		Rcpp::Named("target_error") = pen_err
		);
}

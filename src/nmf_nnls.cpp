#include "nnlm.hpp"


//[[Rcpp::export]]
RcppExport Rcpp::List nmf_nnls(mat A, int k = 1, double eta = 0, double beta = 0, int max_iter = 500, double tol = 1e-5)
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
	 * 	2015-10-28
	 */
	mat W(A.n_rows, k, fill::randu);
	mat H(k, A.n_cols);
	W = normalise(W);  
	mat WtW(W.n_cols, W.n_cols);
	mat HHt(W.n_rows, H.n_rows);
	// err, pen_err = root mean square error/penalized_error
	vec err(max_iter);
	err.fill(-9999);
	vec pen_err(err);
	if (eta < 0) eta = max(max(A));

	int i = 0;
	for(; i < max_iter; i++)
	{
		// solve H given W
		WtW = W.t()*W;
		if (beta > 0) WtW += beta;
		H = nnls_solver(WtW, -W.t()*A, 500*(1+i), tol/(1+i));

		// solve W given H
		HHt = H*H.t();
		if (eta > 0) HHt.diag() += eta;
		W = nnls_solver(HHt, -H*A.t(), 500*(1+i), tol/(1+i)).t();

		pen_err[i] = mean(mean(square(A - W*H)));
		err[i] = sqrt(pen_err[i]);
		if (beta > 0) pen_err[i] += mean(vectorise(square(mean(H)))) * k/A.n_cols;
		if (eta > 0) pen_err[i] += eta * mean(mean(square(W)))*k*k / A.n_rows;
		pen_err[i] = sqrt(pen_err[i]);

	 	if (i > 0 && abs(pen_err[i-1] - pen_err[i]) < tol)
			break;
	}

	if (max_iter <= i)
	{
		Rcpp::Function warning("warning");
		warning("Algorithm does not converge.");
	}

	err.resize(i < max_iter ? i+1 : max_iter);
	pen_err.resize(err.n_elem);

	return Rcpp::List::create(
		Rcpp::Named("W") = W, 
		Rcpp::Named("H") = H, 
		Rcpp::Named("error") = sqrt(err),
		Rcpp::Named("target_error") = sqrt(pen_err),
		Rcpp::Named("eta") = eta,
		Rcpp::Named("beta") = beta
		);
}

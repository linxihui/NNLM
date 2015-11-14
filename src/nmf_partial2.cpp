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
	int k = W.n_cols - H2.n_cols;
	mat Adiff = A;
	Adiff -= W.cols(0, k-1) * H.cols(0, k-1).t();
	if (!W1.empty())
		Adiff -= W1*H.cols(k, H.n_cols-1).t();
	if (!H2.empty())
		Adiff -= W.cols(k, W.n_cols-1)*H2.t();
	return mean(mean(square(Adiff)));
}

mat nnls_solver2(const mat & H, mat mu, const umat & mask, int max_iter, double tol, int n_threads)
{
	mat x(H.n_cols, mu.n_cols, fill::zeros);
	if (n_threads < 0) n_threads = 0; 

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
	for (int j = 0; j < mu.n_cols; j++)
	{
		vec x0(H.n_cols);
		x0.fill(-9999);
		double tmp;
		int i = 0;
		double err1, err2 = 9999;
		//std::cout << "1: " << j << "/" << mu.n_cols << std::endl;
		//std::cout << mask.col(j) << std::endl;;
		do {
			x0 = x.col(j);
			err1 = err2;
			err2 = -1;
			//std::cout << "2: " << j << "/" << mu.n_cols << " : " << i << "/" << max_iter << std::endl;
			for (int k = 0; k < H.n_cols; k++)
			{
				if (mask(k,j)) continue;
				//std::cout << "3: " << j << "/" << mu.n_cols << " : " << i << "/" << max_iter << " : " <<  k << "/" << H.n_cols << " : " << mask(k,j) << std::endl;
				tmp = x.at(k,j) - mu.at(k,j) / H.at(k,k);
				if (tmp < 0) tmp = 0;
				if (tmp != x.at(k,j)) mu.col(j) += (tmp - x.at(k,j)) * H.col(k);
				x.at(k,j) = tmp;
				tmp = std::abs(x.at(k,j) - x0.at(k));
				if (tmp > err2) err2 = tmp;
			}
			//std::cout << "\t" << std::abs(err1 - err2) / (err1 + 1e-9) << endl;
		} while(++i < max_iter && std::abs(err1 - err2) / (err1 + 1e-9) > tol);
	}

	return x;
}


//[[Rcpp::export]]
Rcpp::List nmf_partial2(const mat & A, const mat & W1, const mat & H2, 
	umat Wm, umat Hm, int k, double eta = 0, double beta = 0, 
	int max_iter = 500, double tol = 1e-4, int n_threads = 1, bool show_progress = true, bool show_warning = true)
{
	int n = A.n_rows, m = A.n_cols, kW = W1.n_cols, kH = H2.n_cols;
	int nW = k+kH, nH = k+kW;
	mat W(A.n_rows, nW); //W = [w w2]
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
		umat Wm2(n, kH, fill::zeros);
		Wm = join_horiz(Wm, Wm2);
	}

	mat H(A.n_cols, nH); // H = [h, h1]

	if (!W1.empty())
	{
		umat Hm2(m, kW, fill::zeros);
		Hm = join_horiz(Hm, Hm2);
	}
		
	// err, pen_err = root mean square error/penalized_error
	vec err(max_iter);
	err.fill(-9999);
	vec pen_err(err);

	// check progression
	Progress prgrss(max_iter, show_progress);

	// iterations and tolerance of internal nnls_solver
	int nnls_max_iter = 500;
	double nnls_tol = tol > 1e-4 ? 1e-5 : tol/10.0;

	mat WtW(nH, nH); // known [w w1], get [h h1]
	mat WtA(nH, m);
	mat HtH(nW, nW); // known [h h2], get [w w2]
	mat HtAt(nW, n);

	// solve H = [h h1] given [w w1]
	update_WtW(WtW, W, W1, H2);
	update_WtA(WtA, W, W1, H2, A);

	if (beta > 0) WtW += beta;
	H = nnls_solver2(WtW, WtA, Hm.t(), nnls_max_iter, nnls_tol, n_threads).t();

	prgrss.increment();

	int i = 0;
	for(; i < max_iter; i++)
	{
		Rcpp::checkUserInterrupt();

		prgrss.increment();

		// solve W = [w w2] given [h h2]
		update_WtW(HtH, H, H2, W1);
		update_WtA(HtAt, H, H2, W1, A.t());
		if (eta > 0) HtH.diag() += eta;
		W = nnls_solver2(HtH, HtAt, Wm.t(), nnls_max_iter, nnls_tol, n_threads).t();

		// solve H = [h h1] given [w w1]
		update_WtW(WtW, W, W1, H2);
		update_WtA(WtA, W, W1, H2, A);
		if (beta > 0) WtW += beta;
		H = nnls_solver2(WtW, WtA, Hm.t(), nnls_max_iter, nnls_tol, n_threads).t();

		pen_err[i] = mse(A, W, H, W1, H2);
		err[i] = std::sqrt(pen_err[i]);
		if (beta > 0) pen_err[i] += mean(vectorise(square(mean(H)))) * k/A.n_cols;
		if (eta > 0) pen_err[i] += eta * mean(mean(square(W)))*k*k / A.n_rows;
		pen_err[i] = std::sqrt(pen_err[i]);

		if (i > 0 && std::abs(pen_err[i-1] - pen_err[i]) / (pen_err[i-1] + 1e-6) < tol) 
			break;
	}

	if (max_iter <= i && show_warning)
		Rcpp::warning("Target tolerance not reached. Try a larger max.iter.");

	err.resize(i < max_iter ? i+1 : max_iter);
	pen_err.resize(err.n_elem);

	return Rcpp::List::create(
		Rcpp::Named("W") = W, 
		Rcpp::Named("H") = H.t(), 
		Rcpp::Named("error") = err,
		Rcpp::Named("target_error") = pen_err
		);
}

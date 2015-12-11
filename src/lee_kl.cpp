#include "nnlm.h"

inline int lee_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW, 
	const subview_col<uword> mask, const int & max_iter, const double & rel_tol, const vec & beta)
{
	// Problem:  Aj = W * Hj
	// Method: Lee's mulitiplicative updating when loss = KL divergence 
	// Wt = W^T
	// sumW = column sum of W
	// mask: skip updating
	// beta: a vector of 3, for L2, angle, L1 regularization

	double sumHj = sum(Hj);
	double rel_err = rel_tol + 1;
	double tmp;
	bool is_masked = mask.n_elem > 0;
	vec wh = Wt.t()*Hj;
	int t = 0;
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (int k = 0; k < Wt.n_rows; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			tmp = as_scalar(Wt.row(k)*(Aj/wh));
			tmp /= (sumW(k) + beta(0)*Hj(k) + beta(1)*(sumHj-Hj(k)) + beta(2));
			wh += (tmp-1)*Hj(k) * Wt.row(k).t();
			sumHj += (tmp-1)*Hj(k);
			Hj(k) *= tmp;
			tmp = 2*std::abs(tmp-1)/(tmp+1);
			if (tmp > rel_err) rel_err = tmp;
		}
	}
	return t;
}

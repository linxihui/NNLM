#include "nnlm.h"

int lee_ls_update(subview_col<double> Hj, const mat & WtW, const vec & WtAj, const double & beta3, 
	const subview_col<uword> mask, int max_iter, const double & rel_tol)
{
	// Problem:  Aj = W * Hj
	// Method: Lee's multiplicative update when loss function = square error 
	// WtW = W^T W
	// WtAj = W^T Aj
	// beta3: L1 regularization
	// mask: skip updating
	
	double tmp;
	double rel_err = rel_tol + 1;
	bool is_masked = mask.n_elem > 0;
	int t = 0;
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (int k = 0; k < WtW.n_cols; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			tmp = dot(WtW.col(k), Hj) + beta3;
			tmp = WtAj(k) / tmp;
			Hj(k) *= tmp;
			tmp = 2*std::abs(tmp-1)/(tmp+1);
			if (tmp > rel_err) rel_err = tmp;
		}
	}
	return t;
}

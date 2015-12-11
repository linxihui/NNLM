#include "nnlm.h"

inline int scd_ls_update(subview_col<double> Hj, const mat & WtW, vec & mu, const subview_col<uword> mask, const int & max_iter, const double & rel_tol)
{
	// Problem:  Aj = W * Hj
	// Method: sequential coordinate-wise descent when loss function = square error 
	// WtW = W^T W
	// WtAj = W^T Aj
	// beta3: L1 regularization
	// mask: skip updating

	double tmp;
	double etmp = 0;
	double rel_err = 1 + rel_tol;
	bool is_masked = mask.n_elem > 0;

	int t = 0;
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (int k = 0; k < WtW.n_cols; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			tmp = Hj(k) - mu(k) / WtW(k,k);
			if (tmp < 0) tmp = 0;
			if (tmp != Hj(k))
				mu += (tmp - Hj(k)) * WtW.col(k);
			else
				continue;
			etmp = 2*std::abs(Hj(k)-tmp) / (tmp+Hj(k));
			if (etmp > rel_err)
				rel_err = etmp;
			Hj(k) = tmp;
		}
	}
	return t;
}


#include "nnlm.h"

inline int scd_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW, const subview_col<uword> mask, 
	const int & max_iter, const double & rel_tol, const vec & beta)
{
	// Problem:  Aj = W * Hj
	// Method: Sequentially minimize KL distance using quadratic approximation
	// Wt = W^T
	// sumW = column sum of W
	// mask: skip updating
	// beta: a vector of 3, for L2, angle, L1 regularization

	double sumHj = sum(Hj);
	vec Ajt = Wt.t()*Hj;
	vec mu;
	double a; // 2nd-order-derivative
	double b; // 1st-order-derivative
	double tmp, etmp;
	double rel_err = 1 + rel_tol;
	bool is_masked = mask.n_elem > 0;

	int t = 0; 
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (int k = 0; k < Wt.n_rows; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			mu = Wt.row(k).t()/(Ajt + 1e-16);
			a = dot(Aj, square(mu));
			b = dot(Aj, mu) - sumW(k); // 0.5*ax^2 - bx
			a += beta(0);
			b += a*Hj(k) - beta(2) - beta(1)*(sumHj - Hj(k));
			tmp = b/a; 
			if (tmp < 0) tmp = 0;
			if (tmp != Hj(k))
			{
				Ajt += (tmp - Hj(k)) * Wt.row(k).t();
				etmp = 2*std::abs(Hj(k)-tmp) / (tmp+Hj(k));
				if (etmp > rel_err)
					rel_err = etmp;
				sumHj += tmp - Hj(k);
				Hj(k) = tmp;
			}
		}
	}
	return t;
}

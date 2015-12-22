#include "nnlm.h"

int scd_ls_update(subview_col<double> Hj, const mat & WtW, vec & mu, const subview_col<uword> mask, const unsigned int & max_iter, const double & rel_tol)
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

	unsigned int t = 0;
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (unsigned int k = 0; k < WtW.n_cols; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			tmp = Hj(k) - mu(k) / WtW(k,k);
			if (tmp < 0) tmp = 0;
			if (tmp != Hj(k))
				mu += (tmp - Hj(k)) * WtW.col(k);
			else
				continue;
			etmp = 2*std::abs(Hj(k)-tmp) / (tmp+Hj(k)+TINY_NUM);
			if (etmp > rel_err)
				rel_err = etmp;
			Hj(k) = tmp;
		}
	}
	return int(t);
}


int lee_ls_update(subview_col<double> Hj, const mat & WtW, const vec & WtAj, const double & beta3, 
	const subview_col<uword> mask, const unsigned int & max_iter, const double & rel_tol)
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
	unsigned int t = 0;
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (unsigned int k = 0; k < WtW.n_cols; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			tmp = dot(WtW.col(k), Hj) + beta3;
			tmp = WtAj(k) / (tmp+TINY_NUM);
			Hj(k) *= tmp;
			tmp = 2*std::abs(tmp-1)/(tmp+1);
			if (tmp > rel_err) rel_err = tmp;
		}
	}
	return int(t);
}


int scd_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW, const subview_col<uword> mask,
	const vec & beta, const unsigned int & max_iter, const double & rel_tol)
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

	unsigned int t = 0; 
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (unsigned int k = 0; k < Wt.n_rows; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			mu = Wt.row(k).t()/(Ajt + TINY_NUM);
			a = dot(Aj, square(mu));
			b = dot(Aj, mu) - sumW(k); // 0.5*ax^2 - bx
			a += beta(0);
			b += a*Hj(k) - beta(2) - beta(1)*(sumHj - Hj(k));
			tmp = b/(a+TINY_NUM); 
			if (tmp < 0) tmp = 0;
			if (tmp != Hj(k))
			{
				Ajt += (tmp - Hj(k)) * Wt.row(k).t();
				etmp = 2*std::abs(Hj(k)-tmp) / (tmp+Hj(k) + TINY_NUM);
				if (etmp > rel_err)
					rel_err = etmp;
				sumHj += tmp - Hj(k);
				Hj(k) = tmp;
			}
		}
	}
	return int(t);
}


int lee_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW, 
	const subview_col<uword> mask, const vec & beta, const unsigned int & max_iter, const double & rel_tol)
{
	// Problem:  Aj = W * Hj
	// Method: Lee's multiplicative updating when loss = KL divergence 
	// Wt = W^T
	// sumW = column sum of W
	// mask: skip updating
	// beta: a vector of 3, for L2, angle, L1 regularization

	double sumHj = sum(Hj);
	double rel_err = rel_tol + 1;
	double tmp;
	bool is_masked = mask.n_elem > 0;
	vec wh = Wt.t()*Hj;
	unsigned int t = 0;
	for (; t < max_iter && rel_err > rel_tol; t++)
	{
		rel_err = 0;
		for (unsigned int k = 0; k < Wt.n_rows; k++)
		{
			if (is_masked && mask(k) > 0) continue;
			tmp = as_scalar(Wt.row(k)*(Aj/(wh+TINY_NUM)));
			tmp /= (sumW(k) + beta(0)*Hj(k) + beta(1)*(sumHj-Hj(k)) + beta(2));
			wh += (tmp-1)*Hj(k) * Wt.row(k).t();
			sumHj += (tmp-1)*Hj(k);
			Hj(k) *= tmp;
			tmp = 2*std::abs(tmp-1)/(tmp+1);
			if (tmp > rel_err) rel_err = tmp;
		}
	}
	return int(t);
}

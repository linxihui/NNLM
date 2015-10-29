#' @title Non-negative least square
#' @description Sequential Coordinate-wise algorithm for non-negative least square regression problem
#' @param x       	Design mamtrix
#' @param y       	Vector or matrix of response
#' @param max.iter	Maximum number of iterations
#' @param tol     	Stop criterion, minimum change on x between two successive iteration.
#' @return A vector of non-negative coefficients, a solution to 
#' 	argmin_{beta} ||y - x*beta||_F^2, s.t. beta >= 0
#' @reference http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-TR-2005-06.pdf 
#' @author: Eric Xihui Lin <xihuil.silence@gmail.com>
#' @examples
#'
#' @export
nnls <- function(x, y, max.iter = 500L, tol = 1e-6) {
	if (is.matrix(x)) x <- as.matrix(x);
	if (is.double(x)) storage.mode(x) <- 'double';
	if (is.matrix(y)) y <- as.matrix(y);
	if (is.double(y)) storage.mode(y) <- 'double';

	return(.Call('nnls', x, y, as.integer(max.iter), as.double(tol)));
	}

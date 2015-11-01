#' Non-negative least square
#'
#' Sequential coordinate-wise algorithm for non-negative least square regression
#'
#' @param x             Design mamtrix
#' @param y             Vector or matrix of response
#' @param check.x       If to check the condition number of matrix x to ensure unique solution
#' @param max.iter      Maximum number of iterations
#' @param tol           Stop criterion, minimum change on x between two successive iteration
#' @param n.threads     An integer number of threads/CPUs to use. Default to 0, which depends on OPENMP (usually all cores)
#' @param show.progress TRUE/FALSE indicating if to show a progress bar
#' @return A vector of non-negative coefficients, a solution to 
#' 	argmin_{beta} ||y - x*beta||_F^2, s.t. beta >= 0
#' @references
#' 	Franc, V. C., Hlavac, V. C., Navara, M. (2005). Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem.
#' 	Proc. Int'l Conf. Computer Analysis of Images and Patterns. Lecture Notes in Computer Science 3691. p. 407. \cr
#' 	http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-TR-2005-06.pdf
#' @author Eric Xihui Lin, \email{xihuil.silence@@gmail.com}
#' @examples
#'
#' x <- 10*matrix(runif(50*20), 50, 20)-1
#' beta <- matrix(rexp(20*2), 20, 2)
#' y <- x %*% beta + matrix(rnorm(50*2), 50, 2)
#' beta.hat <- nnls(x, y)
#'
#' @export
#'
nnls <- function(x, y, check.x = TRUE, max.iter = 10000L, tol = .Machine$double.eps, n.threads = 0L, show.progress = TRUE) {
	if (!is.matrix(x)) x <- as.matrix(x);
	if (!is.double(x)) storage.mode(x) <- 'double';
	if (!is.matrix(y)) y <- as.matrix(y);
	if (!is.double(y)) storage.mode(y) <- 'double';

	if (nrow(x) != nrow(y))
		stop("Dimensions of x and y do not match.");
	if (anyNA(x)) stop("x contains missing values.");
	if (anyNA(y)) stop("y contains missing values.");

	if (n.threads < 0L) n.threads <- 0L;

	if (check.x) {
		if (nrow(x) < ncol(x) || rcond(x) < .Machine$double.eps)
			warning("x does not have a full column rank. Solution may not be unique.");
		}

	beta <- .Call('c_nnls', x, y, as.integer(max.iter), as.double(tol), as.integer(n.threads), as.logical(show.progress), PACKAGE = 'NNLM');

	dimnames(beta) <- list(colnames(x), colnames(y));
	if (1 == ncol(beta)) beta <- beta[, 1];

	return(beta);
	}

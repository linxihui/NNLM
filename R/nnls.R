#' Non-negative least square (NNLS)
#'
#' Sequential coordinate-wise algorithm for non-negative least square regression \cr
#' 		\deqn{argmin_{\beta \ge 0} ||y - x*\beta||_F^2}
#'
#' @param x             Design matrix
#' @param y             Vector or matrix of response
#' @param check.x       If to check the condition number of matrix x to ensure unique solution
#' @param max.iter      Maximum number of iterations
#' @param rel.tol       Stop criterion, relative change on x between two successive iteration
#' @param n.threads     An integer number of threads/CPUs to use. Default to 1 (no parallel). Use 0 for all cores
#' @param show.progress TRUE/FALSE indicating if to show a progress bar
#' @return A list of with components \itemize{
#' 	\item{coefficients: } {a matrix or vector (depend on y) of the NNLS solution beta}
#' 	\item{iteration: }{a vector of numbers of iterations for each column of y}
#' 	\item{abs.err: }{a vector of absolute error, i.e., difference between two successive iterations, for each column of y}
#' 	\item{rel.err: }{a vector of relative error, i.e., relative difference between two successive iterations, for each column of y}
#' 	}
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
nnls <- function(x, y, check.x = TRUE, max.iter = 10000L, rel.tol = .Machine$double.eps, n.threads = 1L, show.progress = TRUE) {
	if (!is.matrix(x)) x <- as.matrix(x);
	if (!is.double(x)) storage.mode(x) <- 'double';
	if (is.y.not.matrix <- !is.matrix(y)) y <- as.matrix(y);
	if (!is.double(y)) storage.mode(y) <- 'double';

	if (nrow(x) != nrow(y))
		stop("Dimensions of x and y do not match.");
	if (anyNA(x)) stop("x contains missing values.");
	if (anyNA(y)) stop("y contains missing values.");

	if (max.iter <= 0L) stop("max.iter must be positive.");
	if (n.threads < 0L) n.threads <- 0L; # use all free cores

	if (check.x) {
		if (nrow(x) < ncol(x) || rcond(x) < .Machine$double.eps)
			warning("x does not have a full column rank. Solution may not be unique.");
		}

	# x, y are passed to C++ function by reference (const type)
	sol <- .Call('NNLM_nnls', x, y, as.integer(max.iter), as.double(rel.tol), as.integer(n.threads), as.logical(show.progress), PACKAGE = 'NNLM');
	names(sol) <- c('coefficients', 'iteration', 'abs.err', 'rel.err');

	# check error, the following are mostly due to keyboard interrupt.
	if (any(sol$abs.err < 0) || any(sol$rel.tol < 0) || any(sol$iteration < 0)) {
		warning("Program interrupted. Solution may be problematic.");
		}

	if (!is.null(colnames(x)) || !is.null(colnames(y)))
		dimnames(sol$coefficients) <- list(colnames(x), colnames(y));
	if (is.y.not.matrix)
		sol$coefficients <- sol$coefficients[, seq_len(ncol(sol$coefficients)), drop = TRUE];

	return(sol);
	}

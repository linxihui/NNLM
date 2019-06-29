#' Non-negative linear model/regression (NNLM)
#'
#' Solving non-negative linear regression problem as \cr
#' 		\deqn{argmin_{\beta \ge 0} L(y - x\beta) + \alpha_1 ||\beta||_2^2 +
#' 		\alpha_2 \sum_{i < j} \beta_{\cdot i}^T \beta_{\cdot j}^T + \alpha_3 ||\beta||_1}
#' 	where \eqn{L} is a loss function of either square error or Kullback-Leibler divergence.
#'
#' The linear model is solve in column-by-column manner, which is parallelled. When \eqn{y_{\cdot j}} (j-th column) contains missing values,
#' only the complete entries are used to solve \eqn{\beta_{\cdot j}}. Therefore, the minimum complete entries of each column should be
#' not smaller than number of columns of \code{x} when penalty is not used.
#'
#' \code{method = 'scd'} is recommended, especially when the solution is probably sparse. Though both "mse" and "mkl" loss are supported for
#' non-negative \code{x} and \code{y}, only "mse" is proper when either \code{y} or \code{x} contains negative value. Note that loss "mkl"
#' is much slower then loss "mse", which might be your concern when \code{x} and \code{y} is extremely large.
#'
#' \code{mask} is can be used for hard regularization, i.e., forcing entries to their initial values (if \code{init} specified) or 0 (if
#' \code{init} not specified). Internally, \code{mask} is achieved by skipping the masked entries during the element-wse iteration.
#'
#' @param x             Design matrix
#' @param y             Vector or matrix of response
#' @param alpha         A vector of non-negative value length equal to or less than 3, meaning [L2, angle, L1] regularization on \code{beta}
#'                      (non-masked entries)
#' @param method        Iteration algorithm, either 'scd' for sequential coordinate-wise descent or 'lee' for Lee's multiplicative algorithm
#' @param loss          Loss function to use, either 'mse' for mean square error or 'mkl' for mean KL-divergence. Note that if \code{x}, \code{y}
#'                      contains negative values, one should always use 'mse'
#' @param init          Initial value of \code{beta} for iteration. Either NULL (default) or a non-negative matrix of
#' @param mask          Either NULL (default) or a logical matrix of the same shape as \code{beta}, indicating if an entry should be fixed to its initial
#'                      (if \code{init} specified) or 0 (if \code{init} not specified).
#' @param check.x       If to check the condition number of \code{x} to ensure unique solution. Default to \code{TRUE} but could be slow
#' @param max.iter      Maximum number of iterations
#' @param rel.tol       Stop criterion, relative change on x between two successive iteration. It is equal to \eqn{2*|e2-e1|/(e2+e1)}.
#'                      One could specify a negative number to force an exact \code{max.iter} iteration, i.e., not early stop
#' @param n.threads     An integer number of threads/CPUs to use. Default to 1 (no parallel). Use 0 or a negative value for all cores
#' @param show.warning  If to shown warnings if exists. Default to TRUE
#'
#' @return An object of class 'nnlm', which is a list with components \itemize{
#' 	\item coefficients : a matrix or vector (depend on y) of the NNLM solution, i.e., \eqn{\beta}
#' 	\item n.iteration  : total number of iteration (sum over all column of \code{beta})
#' 	\item error        : a vector of errors/loss as c(MSE, MKL, target.error) of the solution
#' 	\item options      : list of information of input arguments
#' 	\item call         : function call
#' 	}
#'
#' @references
#'
#' Franc, V. C., Hlavac, V. C., Navara, M. (2005). Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem.
#' Proc. Int'l Conf. Computer Analysis of Images and Patterns. Lecture Notes in Computer Science 3691. p. 407.
#'
#' Lee, Daniel D., and H. Sebastian Seung. 1999. "Learning the Parts of Objects by Non-Negative Matrix Factorization."
#' Nature 401: 788-91.
#'
#' Pascual-Montano, Alberto, J.M. Carazo, Kieko Kochi, Dietrich Lehmann, and Roberto D.Pascual-Marqui. 2006.
#' "Nonsmooth Nonnegative Matrix Factorization (NsNMF)." IEEE Transactions on Pattern Analysis and Machine Intelligence 28 (3): 403-14.
#'
#' @author Eric Xihui Lin, \email{xihuil.silence@@gmail.com}
#' @examples
#'
#' # without negative value
#' x <- matrix(runif(50*20), 50, 20);
#' beta <- matrix(rexp(20*2), 20, 2);
#' y <- x %*% beta + 0.1*matrix(runif(50*2), 50, 2);
#' beta.hat <- nnlm(x, y, loss = 'mkl');
#'
#' # with negative values
#' x2 <- 10*matrix(rnorm(50*20), 50, 20);
#' y2 <- x2 %*% beta + 0.2*matrix(rnorm(50*2), 50, 2);
#' beta.hat2 <- nnlm(x, y);
#'
#' @export
nnlm <- function(x, y, alpha = rep(0, 3), method = c('scd', 'lee'),
	loss = c('mse', 'mkl'), init = NULL, mask = NULL, check.x = TRUE,
	max.iter = 10000L, rel.tol = 1e-12, n.threads = 1L,
	show.warning = TRUE) {

	method <- match.arg(method);
	loss <- match.arg(loss);

	if (show.warning && 'mkl' == loss && (any(x < 0) || any(y < 0)))
		warning("x or y have negative values. One should instead use method == 'mse'.");

	is.y.vector <- is.vector(y) && is.atomic(y) && is.numeric(y);
	y <- as.matrix(y);
	check.matrix(y, check.na = FALSE);
	check.matrix(x, check.na = TRUE);

	if (nrow(x) != nrow(y))
		stop("Dimensions of x and y do not match.");

	if (!is.double(x)) storage.mode(x) <- 'double';
	if (!is.double(y)) storage.mode(y) <- 'double';

	if (max.iter <= 0L) stop("max.iter must be positive.");
	if (n.threads < 0L) n.threads <- 0L; # use all free cores

	if (check.x) {
		if (nrow(x) < ncol(x) || rcond(x) < .Machine$double.eps)
			warning("x does not have a full column rank. Solution may not be unique.");
		}

	alpha <- c(alpha, rep(0, 3))[1:3];
	if (show.warning && alpha[1] < alpha[2])
		warning("If alpha[1] < alpha[2], be aware that that algorithm may not converge or unique.");

	check.matrix(mask, dm = c(ncol(x), ncol(y)), mode = 'logical', check.na = TRUE);
	check.matrix(init, dm = c(ncol(x), ncol(y)), check.na = TRUE, check.negative = TRUE);
	if (length(mask) == 0)
		mask <- matrix(FALSE, 0, ncol(y));
	if (length(init) == 0)
		init <- matrix(0.0, 0, ncol(y));
	# if masked but no initialized, masked entries are thought to fix to 0
	if (length(mask) != 0 && length(init) == 0)
		init <- as.double(!mask);
	if (!is.double(init))
		storage.mode(init) <- 'double';

	method.code <- get.method.code(method, loss);

	# x, y are passed to C++ function by reference (const arma::mat & type)
	sol <- c_nnlm(x, y, as.double(alpha), mask, init, as.integer(max.iter),
		as.double(rel.tol), as.integer(n.threads), method.code);

	names(sol) <- c('coefficients', 'n.iteration');
	if (!is.null(colnames(x)))
		rownames(sol$coefficients) <- colnames(x);
	if (!is.null(colnames(y)))
		colnames(sol$coefficients) <- colnames(y);
	if (is.y.vector)
		sol$coefficients <- sol$coefficients[, seq_len(ncol(sol$coefficients)), drop = TRUE];

	error <- mse.mkl(y, x %*% sol$coefficients, na.rm = TRUE, show.warning = FALSE);
	N.complete <- length(y) - sum(is.na(y));

	target.error <- switch(loss,
		'mse' = unname(0.5*error[1]),
		'mkl' = unname(error[2]));

	target.error <- target.error + (alpha[1] - alpha[2])*sum(sol$coefficients^2) +
			alpha[2]*sum(colSums(sol$coefficients)^2) + alpha[3]*sum(sol$coefficients);

	sol$error <- c(error, 'target.error' = target.error);
	sol$options <- list('method' = method, 'loss' = loss, 'max.iter' = max.iter, 'rel.tol' = rel.tol);
	sol$call <- match.call();

	return(structure(sol, class = 'nnlm'));
	}

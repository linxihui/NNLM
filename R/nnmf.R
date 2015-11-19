#' Non-negative matrix factorization
#'
#' Non-negative matrix factorization(NNMF) using alternating NNLS or Brunet's multiplicative update
#'
#' The problem of non-negative matrix factorization is to find \code{W, H, W_1, H_1}, such that \cr
#' 		\deqn{A = W H + W_0 H_1 + W_1 H_0 + noise,}\cr
#' where \eqn{W_0}, \eqn{H_0} are known matrices, which are NULLs in most application case. 
#' In tumour content deconvolution, \eqn{W_0} can be thought as known healthy profile, and \eqn{W}
#' is desired pure cancer profile. One also set \eqn{H_0} to a row matrix of 1, and thus \eqn{W_1}
#' can be treated as common profile among samples.
#'
#' To simplify the notations, we denote right hand side of the above equation as \eqn{W H}. 
#' The problem to solved using square error is\cr
#' 	  \deqn{argmin_{W \ge 0, H \ge 0} ||A - W H||_2^2 + \eta*||W||_F^2 + \beta*\sum_{j=1}^m (||H_j||_1^2),}\cr
#' where \eqn{H_j} is the j-th column of \eqn{H}. This minimization problem is solved by apply 
#' \code{\link{nnls}} to W and H alternatively, when \code{method == 'nnls'},
#' which is implemented using sequential coordinate descend methods. 
#'
#' When \code{method = 'brunet'}, decomposition is done using Brunet's  multiplicative updates based on 
#' Kullback-Leibler divergence. Note that penalties \code{eta}, \code{beta} and known profiles \code{W0} \code{H0} 
#' are not support in this case.
#'
#' @param A             A matrix to be decomposed
#' @param k             An integer of decomposition rank
#' @param method        Decomposition algorithms. Options are 'nnls'(default), 'brunet'
#' @param W0            Known \code{W} (left) profile
#' @param H0            Known \code{H} (right) profile
#' @param W.init        Initial matrix of [W, W1]. Sample frome Unif(0,1) if NULL (default)
#' @param Wm            Masked matrix (logical values) with dimension like [W, W1], s.t. masked entries are fixed to 0. No mask if set to NULL (default)
#' @param Hm            Masked matrix (logical values) with dimension like [H, H1]^T, s.t. masked entries are fixed to 0. No mask if set to NULL (default)
#' @param eta           L2 penalty on the left (W). Default to no penalty. If eta < 0 then eta = max(A). Effective only when \code{method = 'nnls'}
#' @param beta          L1 penalty on the right (H). Default to no penalty. Effective only when \code{method = 'nnls'}
#' @param max.iter      Maximum iteration of alternating NNLS solutions to H and W
#' @param rel.tol       Stop criterion, relative difference of target_error between two successive iterations
#' @param check.k       If to check whether k <= n*m/(n+m), where (n,m)=dim(A)
#' @param n.threads     An integer number of threads/CPUs to use. Default to 1(no parallel). Specify 0 for all cores
#' @param show.progress TRUE/FALSE indicating if to show a progress bar
#' @param show.warning  If to show warnings when targeted \code{rel.tol} is not reached
#' @return A list with components
#' 	\itemize{
#' 		\item{W:}{ left/base matrix W}
#' 		\item{H:}{ right/coefficient matrix H}
#' 		\item{error:}{ root mean square error between A and W*H}
#' 		\item{target.error:}{ error used to stop iteration}
#' 		\item{target.measure:}{ the measure for \code{target.error}}
#' 		\item{H1:}{ coefficient matrix corresponding to known W0}
#' 		\item{W1:}{ base matrix corresponding to known H0}
#' 	} 
#' @author Eric Xihui Lin, \email{xihuil.silence@@gmail.com}
#' @seealso \code{\link{nnls}}, \code{\link{predict.nnmf}}
#' @examples
#'
#' x <- matrix(runif(50*20), 50, 20)
#' r <- nnmf(x, 2)
#'
#' @export
nnmf <- function(
	A, k = 1L, method = c('nnls', 'brunet'), W0 = NULL, H0 = NULL, 
	W.init = NULL, Wm = NULL, Hm = NULL,
	eta = 0, beta = 0, max.iter = 500L, rel.tol = 1e-4, 
	check.k = TRUE, n.threads = 1L, show.progress = TRUE, 
	show.warning = TRUE
	) {
	method = match.arg(method);
	A <- check.input.matrix(A, check.missing = FALSE);
	kW0 <- kH0 <- 0;
	if (!is.null(W0)) {
		W0 <- check.input.matrix(W0);
		if (nrow(W0) != nrow(A)) stop("Rows of A and W0 must match.");
		kW0 <- ncol(W0);
		}
	if (!is.null(H0)) {
		H0 <- check.input.matrix(H0);
		if (ncol(H0) != ncol(A)) stop("Columns of A and H0 must match.");
		kH0 <- nrow(H0);
		}
	if (!is.null(W.init)) {
		W.init <- check.input.matrix(W.init);
		if (any(dim(W.init) != c(nrow(A), k + kW0)))
			stop("Dimension of W.init is invalid.");
		}
	if (!is.null(Wm)) {
		Wm <- check.input.mask(Wm);
		if (ncol(Wm) == k && kW0 > 0)
			Wm <- cbind(Wm, matrix(FALSE, ncol(Wm), kW0));
		if (any(dim(Wm) != c(nrow(A), k + kW0)))
			stop("Dimension of Wm is invalid.");
		}
	if (!is.null(Hm)) {
		Hm <- check.input.mask(Hm);
		if (nrow(Hm) == k && kH0 > 0)
			Hm <- rbind(Hm, matrix(0, kH0, ncol(Hm)));
		if (any(dim(Hm) != c(k + kH0, ncol(A))))
			stop("Dimension of Hm is invalid.");
		}
	if (check.k && k > min(dim(A))) 
		stop("k must not be larger than min(nrow(A), ncol(A))");
	if (eta < 0) eta <- median(A);
	if ('brunet' == method && (!is.null(W0) || !is.null(H0) || !is.null(Wm) || !is.null(Hm) || any(is.na(A))))
		stop("When any of W0, H0, Wm, Hm is not NULL or NA in A, method must be 'nnls'.");

	if (n.threads < 0L) n.threads <- 0L;

	run.time <- system.time(
		out <- switch(method,
			'nnls' = {
				if (!any(is.na(A)) && all(c(is.null(W0), is.null(H0), is.null(Wm), is.null(Hm)))) {
					.Call('NNLM_nmf_nnls', 
						A, as.integer(k), as.double(eta), as.double(beta), as.integer(max.iter), 
						as.double(rel.tol), as.integer(n.threads), as.logical(show.progress), as.logical(show.warning),
						PACKAGE = 'NNLM'
						)
				} else {
					# transform NULLs to empty matrices
					if (is.null(W0)) W0 <- matrix(0., 0, 0);
					if (is.null(H0)) H0 <- matrix(0., 0, 0);
					if (is.null(W.init)) W.init <- matrix(0., 0, 0);
					if (is.null(Wm)) Wm <- matrix(0., 0, 0);
					if (is.null(Hm)) Hm <- matrix(0., 0, 0);
					.Call('NNLM_nnmf_generalized', 
						A, W0, t(H0), W.init, Wm, t(Hm), as.integer(k), as.double(eta), as.double(beta), as.integer(max.iter), 
						as.double(rel.tol), as.integer(n.threads), as.logical(show.progress), as.logical(show.warning), 
						500L, 1e-6, FALSE, PACKAGE = 'NNLM'
						)
					}
				},
			'brunet' = .Call('NNLM_nmf_brunet', 
				A, as.integer(k), as.integer(max.iter), as.double(rel.tol), 
				as.integer(n.threads), as.logical(show.progress), as.logical(show.warning),
				PACKAGE = 'NNLM'
				)
			)
		);
	# add row/col names back
	if (!is.null(rownames(A))) rownames(out$W) <- rownames(A);
	if (!is.null(colnames(A))) colnames(out$H) <- colnames(A);
	if (!is.null(W0)) {
		out$W0 <- W0;
		out$H1 <- out$H[-seq_len(k), ];
		rownames(out$H1) <- colnames(W0);
		out$H <- out$H[seq_len(k), ];
		}
	if (!is.null(H0)) {
		out$W1 <- out$W[, -seq_len(k)];
		colnames(out$W1) <- rownames(W0);
		out$H0 <- H0;
		out$W <- out$W[, seq_len(k)];
		}

	names(out)[4] <- 'target.error';
	out$iteration <- length(out$error);
	out$method <- method;
	out$target.error <- as.vector(out$target.error);
	out$error <- as.vector(out$error);
	out$rel.tol <- abs(diff(tail(out$target.error,2))) / (tail(out$target.error, 1)+1e-6);
	out$target.measure <- ifelse('brunet' == method, 
		'Kullback-Leibler Divergence',
		ifelse(0 == eta && 0 == beta, 'Root Mean Square Error',
			'Penalized Root Mean Square Error'));
	out$system.time <- run.time;

	return(structure(out, class = 'nnmf'));
	}


# check if input is a matrix, non-negative and no-missing
#
# @param A Input matrix to be check
# @return A properly modified matrix
check.input.matrix <- function(A, check.missing = TRUE) {
	input.name <- as.character(substitute(A));
	if (!is.matrix(A)) A <- as.matrix(A);
	if (!is.numeric(A)) stop(sprintf("Matrix %s must be numeric", input.name));
	if (!is.double(A)) storage.mode(A) <- 'double';
	if (any(A < 0)) stop(sprintf("Matrix %s must be non-negative.", input.name));
	if (check.missing && any(is.na(A))) stop(sprintf("Matrix %s contains missing values.", input.name));
	return(A);
	}

# check if input is a mask matrix, non-negative and no-missing
#
# @param A Input matrix to be check
# @return A properly modified matrix
check.input.mask <- function(A) {
	input.name <- as.character(substitute(A));
	if (!is.matrix(A)) A <- as.matrix(A);
	if (!is.logical(A)) storage.mode(A) <- 'logical';
	if (any(is.na(A))) stop(sprintf("Matrix %s contains missing values.", input.name));
	return(A);
	}

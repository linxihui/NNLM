#' Non-negative matrix factorization
#'
#' Non-negative matrix factorization(NNMF) using alternating NNLS or Brunet's multiplicative update
#'
#' The problem of non-negative matrix factorization is to find \code{W, H, W_1, H_1}, such that \cr
#' 		\deqn{A = W H + W_0 H_1 + W_1 H_0 + noise,}\cr
#' where \eqn{W_0}, \eqn{H_0} are known matrices, which are NULLs in most application case. 
#' In tumour content decovolution, \eqn{W_0} can be thought as known healthy profile, and \eqn{W}
#' is desired pure cancer profile. One also set \eqn{H_0} to a row matrix of 1, and thus \eqn{W_1}
#' can be treated as common profile among samples.
#'
#' To simpliyf the notations, we denote right hand side of the above equaiton as \eqn{W H}. 
#' The problem to solved using square error is\cr
#' 	  \deqn{argmin_{W \ge 0, H \ge 0} ||A - W H||_2^2 + \eta*||W||_F^2 + \beta*\sum{j=1}^m (||H_j||_1^2),}\cr
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
#' @param eta           L2 penalty on the left (W). Default to no penalty. If eta < 0 then eta = max(A). Effective only when \code{method = 'nnls'}
#' @param beta          L1 penalty on the right (H). Default to no penalty. Effective only when \code{method = 'nnls'}
#' @param max.iter      Maximum iteration of alternating NNLS solutions to H and W
#' @param rel.tol       Stop criterion, relative difference of target_error between two successive iterations
#' @param check.k       If to check whether k <= n*m/(n+m), where (n,m)=dim(A). Default to TRUE, but it can be slow
#' @param n.threads     An integer number of threads/CPUs to use. Default to 1(no parallel). Specify 0 for all cores
#' @param show.progress TRUE/FALSE indicating if to show a progress bar
#' @param show.warning  If to show warnings when targetted \code{rel.tol} is not reached
#' @return A list with compenents
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
	eta = 0, beta = 0, max.iter = 500L, rel.tol = 1e-4, 
	check.k = TRUE, n.threads = 1L, show.progress = TRUE, 
	show.warning = TRUE
	) {
	method = match.arg(method);
	check.input.matrix(A);
	if (!is.null(W0)) {
		check.input.matrix(W0);
		if (nrow(W0) != nrow(A)) stop("Rows of A and W0 must match.");
		}
	if (!is.null(H0)) {
		check.input.matrix(H0);
		if (ncol(H0) != ncol(A)) stop("Columns of A and H0 must match.");
		}
	if (check.k && k > min(dim(A))) 
		stop("k must not be larger than min(nrow(A), ncol(A))");
	if (eta < 0) eta <- median(A);
	if ('brunet' == method && (!is.null(W0) || !is.null(H0)))
		stop("When W0 or H0 are not NULL, method must be 'nnls'.");
		

	if (n.threads < 0L) n.threads <- 0L;

	run.time <- system.time(
		out <- switch(method,
			'nnls' = {
				if (is.null(W0) && is.null(H0)) {
					.Call('NNLM_nmf_nnls', 
					A, as.integer(k), as.double(eta), as.double(beta), as.integer(max.iter), 
					as.double(rel.tol), as.integer(n.threads), as.logical(show.progress), as.logical(show.warning),
					PACKAGE = 'NNLM'
					)
				} else {
					.Call('NNLM_nmf_partial', 
					A, W0, t(H0), as.integer(k), as.double(eta), as.double(beta), as.integer(max.iter), 
					as.double(rel.tol), as.integer(n.threads), as.logical(show.progress), as.logical(show.warning),
					PACKAGE = 'NNLM'
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
		cout$W <- out$W[, seq_len(k)];
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


# check if input is a matrix of doubles, non-negative and no-missing
#
# @param A Input matrix to be check
# @return NULL
check.input.matrix <- function(A) {
	input.name <- as.character(substitute(A));
	if (!is.matrix(A)) A <- as.matrix(A);
	if (!is.double(A)) storage.mode(A) <- 'double';
	if (any(A < 0)) stop(sprintf("Matrix %s must be non-negative.", input.name));
	if (anyNA(A)) stop(sprintf("Matrix %s contains missing values.", input.name));
	return(NULL);
	}

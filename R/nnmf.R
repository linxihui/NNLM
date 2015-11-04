#' Non-negative matrix factorization
#'
#' Non-negative matrix factorization(NNMF) using alternating NNLS or Brunet's multiplicative update
#'
#' The problem to solved is\cr
#' 	  \deqn{argmin_{W \ge 0, H \ge 0} ||A - WH||_2^2 + \eta*||W||_F^2 + \beta*\sum{j=1}^m (||H.col(j)||_1^2)} \cr
#' If \code{method = 'nnls'}, \code{\link{nnls}} is applied to W and H alternatively, where \code{\link{nnls}} is implemented
#' using sequential coordinate descend methods. If \code{method = 'brunet'}, decomposition is done using Brunet's
#' multiplicative updates based on Kullback-Leibler divergence. Note that penalties \code{eta}, \code{beta} are supported only when \code{method = 'nnls'}.
#'
#' @param A             A matrix to be decomposed
#' @param k             An integer of decomposition rank
#' @param method        Decomposition algorithms. Options are 'nnls'(default), 'brunet'
#' @param eta           L2 penalty on the left (W). Default to no penalty. If eta < 0 then eta = max(A). Effective only when \code{method = 'nnls'}
#' @param beta          L1 penalty on the right (H). Default to no penalty. Effective only when \code{method = 'nnls'}
#' @param max.iter      Maximum iteration of alternating NNLS solutions to H and W
#' @param rel.tol       Stop criterion, relative difference of target_error between two successive iterations
#' @param check.k       If to check whether k <= n*m/(n+m), where (n,m)=dim(A)
#' @param n.threads     An integer number of threads/CPUs to use. Default to 1(no parallel). Specify 0 for all cores
#' @param show.progress TRUE/FALSE indicating if to show a progress bar
#' @return A list of W, H and 
#' 	\itemize{
#' 		\item{error:}{ root mean square error between A and W*H}
#' 		\item{target.error:}{ error used to stop iteration}
#' 		\item{target.measure:}{ the measure for \code{target.error}}
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
	A, k = 1L, method = c('nnls', 'brunet'), eta = 0, beta = 0, max.iter = 500L, 
	rel.tol = 1e-4, check.k = TRUE, n.threads = 1L, show.progress = TRUE
	) {
	method = match.arg(method);
	if (!is.matrix(A)) x <- as.matrix(A);
	if (!is.double(A)) storage.mode(A) <- 'double';
	if (any(A < 0)) stop("Matrix must be non-negative.");
	if (check.k && k > min(dim(A))) 
		stop("k must not be larger than min(nrow(A), ncol(A))");
	if (anyNA(A)) stop("A contains missing values.");
	if (eta < 0) eta <- median(A);

	if (n.threads < 0L) n.threads <- 0L;

	run.time <- system.time(
		out <- switch(method,
			'nnls' = .Call('NNLM_nmf_nnls', 
				A, as.integer(k), as.double(eta), as.double(beta), as.integer(max.iter), 
				as.double(rel.tol), as.integer(n.threads), as.logical(show.progress),
				PACKAGE = 'NNLM'
				),
			'brunet' = .Call('NNLM_nmf_brunet', 
				A, as.integer(k), as.integer(max.iter), as.double(rel.tol), 
				as.integer(n.threads), as.logical(show.progress), 
				PACKAGE = 'NNLM'
				)
			)
		);
	# add row/col names back
	if (!is.null(rownames(A))) rownames(out$W) <- rownames(A);
	if (!is.null(colnames(A))) colnames(out$H) <- colnames(A);

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

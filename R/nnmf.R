#' @title Non-negative matrix factorization
#' @param x       	A matrix to be decomposed
#' @param rank    	An integer of decomposition rank
#' @param method  	Decomposition algorithms. Options are 'nnls'(default), 'brunet'
#' @param eta     	L2 penalty on the left (W). Default to no penalty. If eta < 0 then eta = max(A). Effective only when \code{method = 'nnls'}
#' @param beta    	L1 penalty on the right (H). Default to no penalty. Effective only when \code{method = 'nnls'}
#' @param max.iter	Maximum iteration of alternating NNLS solutions to H and W
#' @param tol     	Stop criterion, maximum difference of target_error between two successive iterations.
#' @return A list of W, H and 
#' 	\itemize{
#' 		\item{error}{root mean square error between A and W*H}
#' 		\item{target_error}{root mean(devided by nxm) square error of the target function. Same as error if no penalty}
#' 	}
#' @details The problem to solved is
#' 	  argmin_{W>=0, H>=0} ||A - WH||_2^2 + eta*||W||_F^2 + beta*sum(||H.col(j)||_1^2)
#' If \code{method = 'nnls'} Apply \code{\link{nnls}} to W and H alternatingly, where \code{\link{nnls}} is implemented 
#' using sequential coordinate descend methods. If \code{method = 'brunet'}, Decomposition is done using Brunet's 
#' multiplicative updates based on KL divergence. Note that penalties \code{eta}, \code{beta} are supported only when \code{method = 'nnls'}.
#' @author Eric Xihui Lin, \email{xihuil.silence@@gmail.com}
#' @seealso \code{\link{nnls}}, \code{\link{predict.NNMF}}
#' @examples
#' x <- matrix(runif(50*20), 50, 20)
#' r <- nnmf(x, 2)
#' @export
nnmf <- function(x, k, method = c('nnls', 'brunet'), eta = 0, beta = 0, max.iter = 500L, tol = 1e-5) {
	method = match.arg(method);
	if (is.matrix(x)) x <- as.matrix(x);
	if (is.double(x)) storage.mode(x) <- 'double';
	out <- switch(method,
		'nnls' = .Call('nmf_nnls', x, as.integer(k), as.double(eta), as.double(beta), as.integer(max.iter), as.double(tol), PACKAGE = "NNLM"),
		'brunet' = .Call('nmf_brunet', x, as.integer(k), as.integer(max.iter), as.double(tol), PACKAGE = "NNLM")
		);
	out$iteration <- length(out$error);
	out$method <- method;
	out$tol <- tol;
	if ('brunet' == method) out$target_error <- out$error;
	return(structure(out, class = 'NNMF'));
	}


#' @title Calculate W or H matrix from a NNMF object given new matrix and pre-computed H or W
#' @param object: An NNMF object returned by \code{\link{nnmf}}
#' @param newdata: A new matrix of x
#' @param which.matrix: Either 'W' or 'H'
#' @return \code{W} or \code{H} for newdata given pre-computed H or W
#' @examples
#' x <- matrix(runif(50*20), 50, 20)
#' r <- nnmf(x, 2)
#' newx <- matrix(runif(30*20), 30, 20)
#' pred <- predict(r, newx, 'H')
#' @seealso \code{\link{nnmf}}
#' @export
predict.NNMF <- function(object, newdata, which.matrix = c('H', 'W'), max.iter = 500L, tol = object$tol) {
	which.matrix <- match.arg(which.matrix);
	if (is.matrix(newdata)) newdata <- as.matrix(newdata);
	if (is.double(newdata)) storage.mode(newdata) <- 'double';
	tol <- as.double(tol);
	max.iter <- as.integer(max.iter);
	
	switch(object$method,
		'nnls' = {
			switch(which.matrix,
				'H' = nnls(object$W, newdata, max.iter, tol),
				'W' = t(nnls(t(object$H), t(newdata), max.iter, tol)))
			},
		'brunet' = {
			switch(which.matrix,
				'H' = .Call('get_H_brunet', newdata, object$W, max.iter, tol, PACKAGE = "NNLM"),
				'W' = t(.Call('get_H_brunet', t(object$H), t(newdata), max.iter, tol, PACKAGE = "NNLM")))
			}
		)
	}

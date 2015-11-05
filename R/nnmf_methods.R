#' Methods for nnmf object returned by \code{nnmf}
#'
#' @param object        An NNMF object returned by \code{\link{nnmf}}
#' @param newdata       A new matrix of x
#' @param which.matrix  Either 'W' or 'H'
#' @param method        Either 'nnls' or 'brunet'. Default to \code{object$method}
#' @param max.iter      Maximum number of iterations
#' @param rel.tol       Stop criterion, relative difference of target_error between two successive iterations
#' @param n.threads     An integer number of threads/CPUs to use. Default to 1(no parallel). Use 0 for all cores
#' @param show.progress TRUE/FALSE indicating if to show a progress bar
#' @param ...           Further arguments passed to 'print', 'plot' or 'heatmap'
#' @return \itemize{
#' 	\item{predict: }{\code{W} or \code{H} for newdata given pre-computed H or W}
#' 	\item{print: }{print running time and accuracy}
#' 	\item{plot: }{line plot if which = 'error' or 'target.error' and heatmap if which = 'H' or 'W'}
#' 	}
#' @examples
#'
#' x <- matrix(runif(50*20), 50, 20)
#' r <- nnmf(x, 2)
#' r
#' newx <- matrix(runif(50*30), 50, 30)
#' pred <- predict(r, newx, 'H')
#' 
#' # check convergence
#' plot(r)
#'
#' # show W matrix
#' plot(r, 'W')
#'
#' @seealso \code{\link{nnmf}}
#' @export
predict.nnmf <- function(
	object, newdata, which.matrix = c('H', 'W'), method = object$method, 
	max.iter = 100L, rel.tol = object$rel.tol, n.threads = 1L, show.progress = TRUE,
	...) {
	which.matrix <- match.arg(which.matrix);
	if (!is.matrix(newdata)) newdata <- as.matrix(newdata);
	if (!is.double(newdata)) storage.mode(newdata) <- 'double';
	if (!all(newdata >= 0)) stop("newdata must be non-negative");
	rel.tol <- as.double(rel.tol);
	max.iter <- as.integer(max.iter);

	method <- match.arg(method, c('nnls', 'brunet'));

	out <- switch(method,
		'nnls' = {
			out <- switch(which.matrix,
				'H' = nnls(object$W, newdata, 
					check.x = TRUE, max.iter = max.iter, rel.tol = rel.tol, 
					n.threads = n.threads, show.progress = show.progress
					)$coefficients,
				'W' = t(nnls(t(object$H), t(newdata), 
						check.x = TRUE, max.iter = max.iter, rel.tol = rel.tol, 
						n.threads = n.threads, show.progress = show.progress
						)$coefficients)
				)
			},
		'brunet' = {
			switch(which.matrix,
				'H' = .Call('NNLM_get_H_brunet', 
					newdata, object$W, max.iter, rel.tol, n.threads, show.progress, 
					PACKAGE = 'NNLM'
					),
				'W' = t(.Call( 'NNLM_get_H_brunet', 
					t(newdata), t(object$H), max.iter, rel.tol, n.threads, show.progress, 
					PACKAGE = 'NNLM')
					)
				)
			}
		);

	if ('H' == which.matrix) colnames(out) <- colnames(newdata);
	if ('W' == which.matrix) rownames(out) <- rownames(newdata);

	return(out);
	}


#' @rdname predict.nnmf
#' @export
print.nnmf <- function(x, ...) {
	print(x$system.time);
	nstep <- length(x$error);
	cat('RMSE: ', x$error[nstep], '\n', sep='');
	switch(x$target.measure,
		'Kullback-Leibler Divergence' = cat('KL divergence: ', x$target.error[nstep], '\n', sep=''),
		'Penalized Root Mean Square Error' = cat('Penalized RMSE: ', x$target.error[nstep], '\n', sep='')
		);
	}

#' @rdname predict.nnmf
#' @param x     An NNMF object returned by \code{\link{nnmf}}
#' @param which One of 'error', 'target', 'W', 'H'. The first two give line plots while the latter two give heatmaps
#' @export
plot.nnmf <- function(x, which = c('error', 'target.error', 'W', 'H'),  ...) {
	which <- match.arg(which);
	dots <- list(...);

	default.setting <- list(
		x[[which]],
		type = 'l', 
		lwd = 2,
		xlab = 'Iteration',
		ylab = which
		);
	dots <- modifyList(default.setting, dots);

	switch(which,
		'error' = ,
		'target.error' = do.call(plot, dots),
		'W' =,
		'H' = heatmap(x[[which]], ...))
	}

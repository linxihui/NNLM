#' Methods for nnmf object returned by \code{nnmf}
#'
#' @param object        An NNMF object returned by \code{\link{nnmf}}
#' @param newdata       A new matrix of x. No required when \code{which == 'A'}
#' @param which         Either 'A' (default), 'W' or 'H'
#' @param method        Either 'scd' or 'lee'. Default to \code{object$options$method}
#' @param loss          Either 'mse' or 'mkl'. Default to \code{object$options$loss}
#' @param x             An NNMF object returned by \code{\link{nnmf}}
#' @param ...           Further arguments passed to 'nnlm' or 'print'
#' @return 'A' or a class of 'nnlm' for 'predict.nnmf' and no return for 'print'.
#'
#' @examples
#'
#' x <- matrix(runif(50*20), 50, 20)
#' r <- nnmf(x, 2)
#' r
#' newx <- matrix(runif(50*30), 50, 30)
#' pred <- predict(r, newx, 'H')
#'
#' @seealso \code{\link{nnmf}}, \code{\link{nnlm}}
#' @export
predict.nnmf <- function(
	object, newdata, which = c('A', 'W', 'H'),
	method = object$options$method,
	loss = object$options$loss,
	...) {

	which <- match.arg(which);
	if (which != 'A') {
		if('W' == which)
			check.matrix(newdata, dm = c(NA, ncol(object$H)));
		if('H' == which)
			check.matrix(newdata, dm = c(nrow(object$W),NA ));
		if (!is.double(newdata))
			storage.mode(newdata) <- 'double';
		}

	out <- switch(which,
		'A' = object$W %*% object$H,
		'W' = nnlm(t(object$H), t(newdata), method = method, loss = loss, ...),
		'H' = nnlm(object$W, newdata, method = method, loss = loss, ...)
		);

	if ('W' == which)
		out$coefficients <- t(out$coefficients);

	return(out);
	}


#' @rdname predict.nnmf
#' @export
print.nnmf <- function(x, ...) {
	if (x$n.iteration < 2) {
		rel.tol <- NA_real_;
	} else {
		err <- tail(x$target.loss, 2);
		rel.tol <- diff(err)/mean(err); 
		}
	cat("Non-negative matrix factorization:\n")
	if (x$options$method == 'scd') {
		cat("   Algorithm: Sequential coordinate-wise descent\n");
	} else {
		cat("   Algorithm: Lee's multiplicative algorithm\n");
		}
	if (x$options$loss == 'mse') {
		cat("        Loss: Mean squared error\n");
	} else {
		cat("        Loss: Mean Kullback-Leibler divergence\n");
		}
		cat("         MSE: ", tail(x$mse, 1), '\n', sep = '');
		cat("         MKL: ", tail(x$mkl, 1), '\n', sep = '');
		cat("      Target: ", tail(x$target.loss, 1), '\n', sep = '');
		cat("   Rel. tol.: ", sprintf("%.3g", abs(rel.tol)), '\n', sep = '');
		cat("Total epochs: ", as.integer(sum(x$average.epochs)), '\n', sep = '');
		cat("# Interation: ", x$n.iteration, '\n', sep = '');
		cat("Running time:\n");
	print(x$run.time);
	invisible(NULL);
	}

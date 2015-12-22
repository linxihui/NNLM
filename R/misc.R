#' Compute mean square error(MSE) and mean kL divergence (MKL)
#' 
#' @param obs          observed value
#' @param pred         prediction/estimate
#' @param na.rm        if to remove NAs
#' @param show.warning if to show warning if any
#' @return A vector of c(MSE, MKL)
#' @export
mse.mkl <- function(obs, pred, na.rm = TRUE, show.warning = TRUE) {
	mkl <- ifelse(
		!show.warning && (any(obs < 0) || any(pred < 0)), NaN, 
		mean((obs + 1e-16)*log((obs+1e-16) / (pred+1e-16)) - obs + pred, na.rm = na.rm)
		);
	mse <- mean((obs - pred)^2, na.rm = na.rm)
	return(c(MSE = mse, MKL = mkl));
	}

# Get method code passed to C++ functions
# 	1 = "scd" + "mse"
# 	2 = "lee" + "mse"
# 	3 = "scd" + "mkl"
# 	4 = "lee" + "mkl"
#
# @param method Either sequential coordinate-wise descent (SCD) or Lee's multiplicative algorithm
# @param loss Loss function, either mean square error (MSE) or mean KL-divergence (MKL)
# @return method code from 1L - 4L
#
get.method.code <- function(method = c('scd', 'lee'), loss = c('mse', 'mkl')) {
	method <- match.arg(method);
	loss <- match.arg(loss);
	code <- 1L;
	if ('mkl' == loss) code <- code + 2L;
	if ('lee' == method) code <- code + 1L;
	return(code);
	}


check.matrix <- function(A, dm = NULL, mode = 'numeric', check.na = FALSE, input.name = '', check.negative = FALSE) {
	if (is.null(A)) return(invisible(NULL));
	if (!is.null(dm) && any(dim(A) != dm, na.rm = TRUE))
		stop(sprintf("Dimension of matrix %s is expected to be (%d, %d), but got (%d, %d)", input.name, nrow(A), ncol(A), dm[1], dm[2]));
	if (mode(A) != mode) stop(sprintf("Matrix %s must be %s.", input.name, mode));
	if (check.negative && any(A[!is.na(A)] < 0)) stop(sprintf("Matrix %s must be non-negative.", input.name));
	if (check.na && any(is.na(A))) stop(sprintf("Matrix %s contains missing values.", input.name));
	}


reformat.input <- function(init, mask, n, m, k) {
	if (is.null(mask)) mask <- list();
	if (is.null(init)) init <- list();
	stopifnot(is.list(mask));
	stopifnot(is.list(init));

	known.W <- !is.null(init[['W0']]);
	known.H <- !is.null(init[['H0']]);
	kW0 <- kH0 <- 0;

	is.empty <- function(x) 0 == length(x);

	if (known.W) {
		if(!is.matrix(init[['W0']]))
			init[['W0']] <- as.matrix(init[['W0']]);
		kW0 <- ncol(init[['W0']]);
		mask[['W0']] <- matrix(TRUE, n, kW0);
		}
	else {
		mask[['W0']] <- NULL;
		mask[['H1']] <- NULL;
		init[['H1']] <- NULL;
		}

	if (known.H) {
		if(!is.matrix(init$H0))
			init[['H0']] <- as.matrix(init[['H0']]);
		kH0 <- nrow(init[['H0']]);
		mask[['H0']] <- matrix(TRUE, kH0, m);
		}
	else {
		mask[['H0']] <- NULL;
		mask[['W1']] <- NULL;
		init[['W1']] <- NULL;
		}

	K <- k + kW0 + kH0;

	ew <- !all(sapply(mask[c('W', 'W0', 'W1')], is.empty));
	eh <- !all(sapply(mask[c('H', 'H0', 'H1')], is.empty));
	dim.mask <- list(
		'W' = c(n, k*ew), 'W0' = c(n, kW0*ew), 'W1' = c(n, kH0*ew), 
		'H' = c(k*eh, m), 'H1' = c(kW0*eh, m), 'H0' = c(kH0*eh, m)
		);

	for (mat in c('W', 'W0', 'W1', 'H' ,'H0', 'H1')) {
		check.matrix(mask[[mat]], dim.mask[[mat]], 'logical', TRUE, paste0('mask$', mat));
		if (is.empty(mask[[mat]]))
			mask[[mat]] <- matrix(FALSE, dim.mask[[mat]][[1]], dim.mask[[mat]][[2]]);
		}

	ew <- !all(sapply(init[c('W', 'W0', 'W1')], is.empty));
	eh <- !all(sapply(init[c('H', 'H0', 'H1')], is.empty));
	dim.init <- list(
		'W' = c(n, k*ew), 'W0' = c(n, kW0*ew), 'W1' = c(n, kH0*ew), 
		'H' = c(k*eh, m), 'H1' = c(kW0*eh, m), 'H0' = c(kH0*eh, m)
		);
	for (mat in c('W', 'W0', 'W1', 'H' ,'H0', 'H1')) {
		check.matrix(init[[mat]], dim.init[[mat]], 'numeric', TRUE, paste0('init$', mat));
		if (is.empty(init[[mat]])) {
			init[[mat]] <- matrix(
				runif(prod(dim.init[[mat]])),
				dim.init[[mat]][[1]],
				dim.init[[mat]][[2]]
				);
			#init[[mat]][mask[[mat]]] <- 0;
			}
		if (!is.double(init[[mat]]))
			storage.mode(init[[mat]]) <- 'double';
		}

	return(
		list(
			Wm = do.call(cbind, mask[c('W', 'W0', 'W1')]),
			Hm = do.call(rbind, mask[c('H', 'H1', 'H0')]),
			Wi = do.call(cbind, init[c('W', 'W0', 'W1')]),
			Hi = do.call(rbind, init[c('H', 'H1', 'H0')]),
			kW0 = kW0,
			kH0 = kH0,
			K = K
		));
	}

context("Test non-negative matrix factorization.");

test_that("Test NMF using nnls", {
  suppressWarnings(RNGversion("3.5.0"));
	n <- 50; m <- 10;
	k <- 3; k1 <- 2; k2 <- 1;

	set.seed(234);
	W <- matrix(runif(n*k), n, k);
	H <- matrix(runif(k*m), k, m);
	A <- W %*% H;

	set.seed(123);
	A.nnmf <- nnmf(A, k, max.iter = 10000L, rel.tol=1e-8);
	A.nnmf.mkl <- nnmf(A, k, max.iter = 2000L, rel.tol = 1e-8, loss='mkl');
	A.nnmf.lee <- nnmf(A, k, max.iter = 10000L, rel.tol=1e-8, method = 'lee');
	A.nnmf.lee.mkl <- nnmf(A, k, method = 'lee', loss='mkl', max.iter = 10000L, rel.tol = 1e-6);

	expect_true(all(A.nnmf$W >= 0));
	expect_true(all(A.nnmf$H >= 0));
	expect_equivalent(with(A.nnmf, W%*%H), A);
	expect_equal(with(A.nnmf.mkl, W%*%H), A, tolerance = 1e-6);
	expect_equal(with(A.nnmf.lee, W%*%H), A, tolerance = 1e-6);
	expect_equal(with(A.nnmf.lee.mkl, W%*%H), A, tolerance = 1e-3);
	expect_equal(dimnames(A.nnmf$W), NULL);
	expect_equal(dimnames(A.nnmf$H), NULL);


	# regularization
	A.nnmf.reg <- nnmf(A, k, max.iter = 10000L, rel.tol=1e-8, alpha = c(0.02, 0.01), beta = c(0, 0, 0.01));
	angle <- function(x, y = x) {
		x <- x %*% diag(1/sqrt(colSums(x^2)+1e-20));
		y <- y %*% diag(1/sqrt(colSums(y^2)+1e-20));
		t(x) %*% y;
		}

	angle(W)
	angle(A.nnmf.reg$W, W)

	# known profiles
	W1 <- matrix(runif(n*k1), n, k1);
	H1 <- matrix(runif(k1*m), k1, m);
	W2 <- matrix(runif(n*k2), n, k2);
	H2 <- matrix(1, k2, m);

	A2 <- A + W1 %*% H1 + W2 %*% H2;
	A2.nnmf <- nnmf(A2, k, init = list(W0 = W1, H0 = H2), max.iter = 1000L, rel.tol=1e-3, inner.max.iter = 20L);
	#expect_true(max(abs(cor(t(H1), t(A2.nnmf$H[seq(k+1, length.out = k1), ])) - cor(t(H1)))) < 0.05);
	#expect_true(abs(cor(W2, A2.nnmf$W[, seq(k+k1+1, length.out = k2), drop = FALSE]) - 1) < 0.05);


	dimnames(A) <- list(paste0('R', 1:nrow(A)), paste0('C', 1:ncol(A)));
	A.nnmf2 <- nnmf(A, 2, alpha = 0.1, beta = 0.01);
	print(A.nnmf2)
	W.new <- predict(A.nnmf2, A[1:4, ], which = 'W')

	expect_warning(nnmf(A, 2, alpha = 0.1, beta = 0, max.iter = 10L),
		'Target tolerance not reached. Try a larger max.iter.');

	expect_error(nnmf(A, 20));

	expect_equal(dimnames(A.nnmf2$W), list(rownames(A), NULL));
	expect_equal(dimnames(A.nnmf2$H), list(NULL, colnames(A)));
	expect_equal(dimnames(W.new$coef), list(rownames(A[1:4, ]), NULL));


	# mask
	set.seed(987);
	W <- matrix(runif(n*k), n, k);
	H <- matrix(runif(k*m), k, m);
	Wm <- matrix(as.logical(rbinom(n*k, 1, 0.2)), n, k);
	Hm <- matrix(as.logical(rbinom(n*k, 1, 0.1)), k, m);
	W[Wm] <- 0;
	H[Hm] <- 0;
	A <- W %*% H;
	A[1, 1] <- NA;

	set.seed(123);
	A.nnmf <- nnmf(A, k, mask=list(W = Wm, H = Hm), max.iter = 10000L, rel.tol=1e-8);

	expect_true(all(A.nnmf$W >= 0));
	expect_true(all(A.nnmf$H >= 0));
	expect_equal(with(A.nnmf, W%*%H), W %*% H);
	expect_true(all(A.nnmf$W[Wm] == 0));
	expect_true(all(A.nnmf$H[Hm] == 0));

	# missing values
	A2 <- A;
	ind <- sample(length(A), length(A)*0.1);
	A2[ind] <- NA;
	set.seed(567);
	A2.miss <- nnmf(A2, k, max.iter = 10000L, rel.tol=1e-8);
	expect_equivalent(with(A2.miss, W%*%H)[ind], A[ind])
	})

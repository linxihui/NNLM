context("Test non-negative matrix factorization.");

test_that("Test NMF using nnls", {
	set.seed(987);
	W <- matrix(runif(90), 30);
	H <- matrix(runif(15), 3);

	A <- W %*% H;
	A.nnmf <- nnmf(A, 3, max.it = 5000L, tol = 1e-16);

	expect_true(all(A.nnmf$W >= 0));
	expect_true(all(A.nnmf$H >= 0));
	expect_equal(with(A.nnmf, W%*%H), A);
	expect_equal(dimnames(A.nnmf$W), NULL);
	expect_equal(dimnames(A.nnmf$H), NULL);

	dimnames(A) <- list(paste0('R', 1:nrow(A)), paste0('C', 1:ncol(A)));

	A.nnmf2 <- nnmf(A, 2, eta = -1, beta = 0.01);
	W.new <- predict(A.nnmf2, A[1:4, ], which = 'W')

	expect_warning(nnmf(A, 2, eta = 0.1, beta = 0, max.it = 10L), 
		'Target tolerence not reached. Try a larger max.iter.');
	expect_error(nnmf(A, 10));

	expect_equal(dimnames(A.nnmf2$W), list(rownames(A), NULL));
	expect_equal(dimnames(A.nnmf2$H), list(NULL, colnames(A)));
	expect_equal(dimnames(W.new), list(rownames(A[1:4, ]), NULL));
	})


test_that("Test NMF using Brunet' multiplicative update", {
	set.seed(987);
	W <- matrix(runif(18), 6);
	H <- matrix(runif(15), 3);

	A <- W %*% H;
	A.nnmf <- nnmf(A, 3, method = 'b', max.it = 5000L, tol = 1e-16);

	expect_true(all(A.nnmf$W >= 0));
	expect_true(all(A.nnmf$H >= 0));
	expect_equal(with(A.nnmf, W%*%H), A);
	expect_equal(dimnames(A.nnmf$W), NULL);
	expect_equal(dimnames(A.nnmf$H), NULL);

	dimnames(A) <- list(paste0('R', 1:6), paste0('C', 1:5));
	A.nnmf2 <- nnmf(A, 2, 'brunet');
	W.new <- predict(A.nnmf2, A[1:4, ], which = 'W')

	expect_equal(dimnames(A.nnmf2$W), list(rownames(A), NULL));
	expect_equal(dimnames(A.nnmf2$H), list(NULL, colnames(A)));
	expect_equal(dimnames(W.new), list(rownames(A[1:4, ]), NULL));

	expect_warning(nnmf(A, 2, 'b', max.it = 5L),
		'Target tolerence not reached. Try a larger max.iter.');

	expect_warning(predict(A.nnmf2, A[, 1:2], max.it = 2, which = 'H'),
		'Target tolerence not reached. Try a larger max.iter.');
	})

context("Test non-negative linear model");

test_that("Testing nnlm result", {

	# case 1: A x = b, where b is a vector
  A <- matrix(c(1.883, 1.237, 0.274, 1.916, 0.807,
                0.375, 2.135, 3.237, 0.706, 0.056,
                3.405, 0.874, 1.511, 1.162, 4.325,
                1.843, 0.751, 0.099, 0.126, 0.208,
                0.133, 0.738, 0.378, 0.741, 0.96,
                2.101, 2.155, 0.481, 2.187, 0.165), ncol=5);
	b <- c(1:4, 0);
	y <- A %*% b;
	sol <- nnlm(A, y)$coefficients;
	expect_equal(c(sol), b);
	expect_named(sol, NULL);

	# case 2: A x = b, where b is a matrix
	rownames(A) <- paste0('R', 1:6)
	colnames(A) <- paste0('C', 1:5)
	b2 = matrix(c(1, 0, 2, 4, 0, 8, 0, 3, 6, 2), ncol=2)
	colnames(b2) <- LETTERS[1:2]
	y2 <- A %*% b2;
	sol2 <- nnlm(A, y2)$coefficients;
	expect_equal(dimnames(sol2), list(colnames(A), colnames(b2)))
	expect_equivalent(sol2, b2);

	# case 3: not unexact non-positive solution
	A2 = matrix(c(0.735, -1.428, 0.619, -0.006, -0.686, -0.279, -0.783, -0.779,
	       -0.375, -0.319, 0.085, -0.768, -0.626, -0.901, 0.664, 0.3,
	       0.075, 0.206, -0.489, -0.628, -0.047, 0.163, 1.292, -0.464,
	       0.305, -0.084, 0.41, 0.184, 1.779, 0.038, 1.176, -0.559,
	       -0.946, -0.665, 0.452, 0.527, -0.23, 1.397, 1.764, 0.486), ncol=5)
	b3 = c(1, -3, 2, 0, 4)
	y3 <- A2 %*% b3;
	sol3 <- nnlm(A2, y3)$coefficients[,1];

	expect_false(all(abs(sol3 - b3) < 1e-6)); # unequal

	expect_equal(
		as.vector(sol3),  # the following results are from nnls::nnls
		c(0.649015454583225, 0, 0.338999499138442, 0.810422082985878, 3.94571883712895)
		)
	});




test_that("Testing nnlm error message", {
  suppressWarnings(RNGversion("3.5.0"));
	set.seed(123);
	A <- matrix(runif(20), 5, 4);
	y <- runif(4);
	expect_error(nnlm(A, y), "Dimensions of x and y do not match.");
	#expect_error(nnls(A, c(y, NA)), "y contains missing values.");
	A[3,2] <- NA;
	#expect_error(nnls(A, c(y, 1)), "x contains missing values.");

	A[3,2] <- 0.3;
	A2 <- t(A);
	expect_warning(nnlm(A2, as.matrix(1:4)), "x does not have a full column rank. Solution may not be unique.");
	});

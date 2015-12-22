context("Test non-negative linear model");

test_that("Testing nnlm result", {
	set.seed(123);

	# case 1: A x = b, where b is a vector
	A <- matrix(rexp(80), 10);
	b <- c(1:7, 0);
	y <- A %*% b;
	sol <- nnlm(A, y)$coefficients;
	expect_equal(c(sol), b);
	expect_named(sol, NULL);

	# case 2: A x = b, where b is a matrix
	A2 = matrix(rnorm(50*20), 50, 20, dimnames = list(paste0('R', 1:50), paste0('C', 1:20)));
	b2 = matrix(runif(20*10), 20) * matrix(rbinom(20*10, 1, 0.7), 20);
	colnames(b2) <- LETTERS[1:10]
	y2 <- A2 %*% b2;
	sol2 <- nnlm(A2, y2)$coefficients;
	expect_equal(dimnames(sol2), list(paste0('C', 1:20), LETTERS[1:10]))
	expect_equivalent(sol2, b2);

	# case 3: not unexact non-positive solution
	set.seed(124);
	A2 = matrix(rnorm(50*10), 50);
	b3 = sample(10);  b3[c(2, 6, 10)] <- c(-3, -1, 0); 
	y3 <- A2 %*% b3;
	sol3 <- nnlm(A2, y3)$coefficients[,1];

	expect_false(all(abs(sol3 - b3) < 1e-6)); # unequal
	
	expect_equal(
		as.vector(sol3),  # the following results are from nnls::nnls
		c(4.46566069809037, 0, 9.42218724878907, 0.831624281462219, 6.03605626970417,
			0, 2.85991832609048, 4.31815165158204, 8.79898220810618, 0.311298663332872)
		)
	});


test_that("Testing nnlm error message", {
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

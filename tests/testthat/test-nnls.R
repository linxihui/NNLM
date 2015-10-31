context("Test non-negative linear model");

test_that("Testing nnlm result", {
	set.seed(123);

	# case 1: A x = b, where b is a vector
	A <- matrix(rexp(80), 10);
	b <- c(1:7, 0);
	y <- A %*% b;
	sol <- nnls(A, y)
	expect_equal(sol, b);
	expect_named(sol, NULL);

	# case 2: A x = b, where b is a matrix
	A2 = matrix(rnorm(50*20), 50, 20, dimnames = list(paste0('R', 1:50), paste0('C', 1:20)));
	b2 = matrix(runif(20*10), 20) * matrix(rbinom(20*10, 1, 0.7), 20);
	colnames(b2) <- LETTERS[1:10]
	y2 <- A2 %*% b2;
	A2 <- as.data.frame(A2);
	sol2 <- nnls(A2, y2);
	expect_equal(dimnames(sol2), list(paste0('C', 1:20), LETTERS[1:10]))
	expect_equivalent(sol2, b2);

	# case 3: not unexact non-positive solution
	A2 = matrix(rnorm(50*10), 50);
	b3 = sample(10);  b3[c(2, 6, 10)] <- c(-3, -1, 0); 
	y3 <- A2 %*% b3;
	sol3 <- nnls(A2, y3)

	expect_false(all(abs(sol3 - b3) < 1e-6)); # unequal
	
	expect_equal(
		sol3,  # the following results are from nnls::nnls
		c(1.44975472845781, 0, 0.669424251909988, 9.8563163005591, 5.75162245057833,
			0, 8.65324047879355, 4.70365568743988, 7.28665465046082, 0.197398079169052)
		);
	
	});


test_that("Testing nnls error message", {
	set.seed(123);
	A <- matrix(runif(20), 5, 4);
	y <- runif(4);
	expect_error(nnls(A, y), "Dimensions of x and y do not match.");
	expect_error(nnls(A, c(y, NA)), "y contains missing values.");
	A[3,2] <- NA;
	expect_error(nnls(A, c(y, 1)), "x contains missing values.");

	A[3,2] <- 0.3;
	A2 <- t(A);
	expect_warning(nnls(A2, 1:4), "x does not have a full column rank. Solution may not be unique.");
	A2 <- cbind(A[, 1:3], A[, 1]+A[, 2]);
	expect_warning(nnls(A2, 1:5), "x does not have a full column rank. Solution may not be unique.");
	});

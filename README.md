# NNLM

[![Build Status](https://api.travis-ci.org/linxihui/NNLM.png?branch=master)](https://travis-ci.org/linxihui/NNLM)
[![Coverage Status](http://codecov.io/github/linxihui/NNLM/coverage.svg?branch=master)](http://codecov.io/github/linxihui/NNLM?branch=master)

This is a package for Non-Negative Linear Models. It implements a
fast sequential coordinate descent algorithm (`nnls`) for non-negative least square (NNLS)
and two fast algorithms for non-negative matrix factorization(`nnmf`).

The function `nnls` in R package [`nnls`](https://cran.r-project.org/web/packages/nnls/index.html)
implemented Lawson-Hanson algorithm in Fortran for the above NNLS problem.
However the Lawson-Hanson algorithm is too slow to be embedded to solve other problems like NMF.
The `nnls` function in this package is implemented in C++, using a coordinate-wise descent algorithm,
which has been shown to be much faster.  `nnmf` is a non-negative matrix factorization solver
using alternating NNLS and Brunet's multiplicative updates,
which are both implemented in C++ too. Due to the fast `nnls`, `nnmf` is way faster
than the standard R package [`NMF`](https://cran.r-project.org/web/packages/NMF/index.html). 
Thus `NNLM` is a package more suitable for larger data sets and bigger hidden features (rank).  
In addition. `nnls` is parallelled via openMP for even better performance.


This package includes two main functions, `nnls` and `nnmf`.  `nnls` solves the following non-negative least square(NNLS)
<p align="center">
argmin||y - x β||₂, s.t., β > 0
</p>
where subscript 2 indicates the Frobenius normal of a matrix, analogous to the L₂ normal of a vector. 
While `nnmf` solves a non-negative
matrix factorization problem like
<p align="center">
argmin ||A - WH||₂² + η ||W||₂² + β Σ_{j=1}^m ||h_j||₁², s.t. W ≥ 0, H ≥ 0
</p>
where `m` is the number of columns of `A`, `h_j` is the j-th column of `H`. Here `η` can used to 
control magnitude of `W` and `β` is for both magnitude and sparsity of matrix `H`.


# Install

```r
library(devtools)
install_github('linxihui/NNLM')
```



# A simple example: Non-small Cell Lung Cancer expression data

```r
library(NNLM);

data(nsclc, package = 'NNLM')
str(nsclc)
```

```
##  num [1:200, 1:100] 7.06 6.41 7.4 9.38 5.74 ...
##  - attr(*, "dimnames")=List of 2
##   ..$ : chr [1:200] "PTK2B" "CTNS" "POLE" "NIPSNAP1" ...
##   ..$ : chr [1:100] "P001" "P002" "P003" "P004" ...
```

```r
# create 5 meta-gene signatures, using only 1 thread (no parallel)
decomp <- nnmf(nsclc[, 1:80], 5, method = 'nnls', n.threads = 1, rel.tol = 1e-6)
decomp
```

```
##    user  system elapsed 
##   5.747   4.420   6.377 
## RMSE: 0.7334227
```

```r
plot(decomp, 'W', xlab = 'Meta-gene', ylab = 'Gene')
plot(decomp, 'H', ylab = 'Meta-gene', xlab = 'Patient')
```

![](https://raw.githubusercontent.com/linxihui/Misc/master/Images/NNLM/nsclc-1.png) 
![](https://raw.githubusercontent.com/linxihui/Misc/master/Images/NNLM/nsclc-2.png) 


```r
plot(decomp, ylab = 'RMSE')
```

<img src="https://raw.githubusercontent.com/linxihui/Misc/master/Images/NNLM/nsclc2-1.png" title="" alt="" style="display: block; margin: auto;" />

We see that the default alternating NNLS method coverage fairly quickly.


```r
# find the expressions of meta-genes for patient 81-100
newH <- predict(decomp, nsclc[, 81:100], which = 'H', show.progress = FALSE)
str(newH)
```

```
##  num [1:5, 1:20] 10 16.2 32.4 21.4 28.4 ...
##  - attr(*, "dimnames")=List of 2
##   ..$ : NULL
##   ..$ : chr [1:20] "P081" "P082" "P083" "P084" ...
```

## Example 2: simulated deconvolution

In micro-array data, the mRNA profile (tumour profile) is typically a mixture of 
cancer specific profile and healthy profile. In NMF, it can be viewed as
<p align="center">
A = W H + W₀ H₁,
</p>
where `W` is unknown cancer profile, and `W₀` is known healthy profile. The task here is
to de-convolute `W`, `H` and `H₁` from `A` and `W₀`. 


A more general deconvolution task can be expressed as
<p align="center">
A = W H + W₀ H₁ + W₁ H₀,
</p>
where `H₀` is known coefficient matrix, e.g. a column matrix of 1. In this scenario,
`W₁` can be interpreted as _homogeneous_ cancer profile within the specific cancer patients,
and `W` is _heterogeneous_ cancer profile of interest for downstream analysis, such as
diagnostic or prognostic capacity, sub-type clustering.


This general deconvolution is implemented in `nnmf` via the alternating NNLS algorithm. 
The known profile `W₀` and `H₀` can be passed via arguments `W0` and `H0`. `L₂` and `L₁`
constrain for unknown matrices are also supported.


```r
# set up matrix
n <- 1000; m <- 200;
k <- 5; k1 <- 2; k2 <- 1;

set.seed(123);
W <- matrix(runif(n*k), n, k); # unknown heterogeneous cancer profile
H <- matrix(runif(k*m), k, m);
W0 <- matrix(runif(n*k1), n, k1); # known healthy profile
H1 <- matrix(runif(k1*m), k1, m);
W1 <- matrix(runif(n*k2), n, k2); # unknown common cancer profile
H0 <- matrix(1, k2, m);
noise <- 0.01*matrix(runif(n*m), n, m);

# A is the observed profile to be de-convoluted
A <- W %*% H + W0 %*% H1 + W1 %*% H0 + noise;

deconvol <- nnmf(A, k = 5, W0 = W0, H0 = H0);
```

```
## Warning in system.time(out <- switch(method, nnls = {: Target tolerence not
## reached. Try a larger max.iter.
```

Check if `W` and `H`, our main interest, are recovered.


```r
round(cor(W, deconvol$W), 2);
```

```
##       [,1]  [,2]  [,3]  [,4]  [,5]
## [1,] -0.01 -0.03  1.00  0.00  0.08
## [2,]  0.98  0.06 -0.05  0.00 -0.05
## [3,]  0.21 -0.08  0.06 -0.17  0.99
## [4,] -0.01  1.00  0.00  0.05 -0.04
## [5,] -0.07  0.02 -0.05  0.99  0.04
```

```r
round(cor(t(H), t(deconvol$H)), 2);
```

```
##       [,1]  [,2]  [,3]  [,4]  [,5]
## [1,]  0.03  0.02  1.00  0.17  0.09
## [2,]  0.99  0.02 -0.01  0.15 -0.02
## [3,]  0.22 -0.02  0.04 -0.06  0.98
## [4,] -0.02  1.00  0.05  0.01  0.03
## [5,]  0.09 -0.01  0.11  1.00  0.11
```

We see that `W`, `H` are just permuted. However, as we known that
the minimization problem for NMF usually has not unique solutions
for `W` and `H`. Therefore, `W` and `H` cannot be guaranteed to
be recovered exactly(different only with a permutation and a scaling).


```r
permutation <- c(3, 1, 5, 2, 4);
round(cor(W, deconvol$W[, permutation]), 2);
```

```
##       [,1]  [,2]  [,3]  [,4]  [,5]
## [1,]  1.00 -0.01  0.08 -0.03  0.00
## [2,] -0.05  0.98 -0.05  0.06  0.00
## [3,]  0.06  0.21  0.99 -0.08 -0.17
## [4,]  0.00 -0.01 -0.04  1.00  0.05
## [5,] -0.05 -0.07  0.04  0.02  0.99
```

```r
round(cor(t(H), t(deconvol$H[permutation, ])), 2);
```

```
##       [,1]  [,2]  [,3]  [,4]  [,5]
## [1,]  1.00  0.03  0.09  0.02  0.17
## [2,] -0.01  0.99 -0.02  0.02  0.15
## [3,]  0.04  0.22  0.98 -0.02 -0.06
## [4,]  0.05 -0.02  0.03  1.00  0.01
## [5,]  0.11  0.09  0.11 -0.01  1.00
```


As from the following result, `H₁`, coefficients of health profile and
`W₁`, common cancer profile, are recovered fairly well.


```r
round(cor(t(H1)), 2);
```

```
##      [,1] [,2]
## [1,] 1.00 0.16
## [2,] 0.16 1.00
```

```r
round(cor(t(H1), t(deconvol$H1)), 2);
```

```
##      [,1] [,2]
## [1,] 1.00 0.15
## [2,] 0.16 1.00
```

```r
round(cor(W1, deconvol$W1), 2);
```

```
##      [,1]
## [1,]    1
```


# TODO
1. Add support for meta-genes: thresholding
2. ~~Heatmap~~
3. ~~Examples~~
4. ~~Vignette~~
5. ~~Test~~
6. ~~.traivs.yml~~
7. ~~code coverage~~
8. ~~Parallel, openMP support~~
9. Support for missing values in NMF (can be used for imputation)

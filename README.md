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

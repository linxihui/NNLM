# NNLM

[![Build Status](https://api.travis-ci.org/linxihui/NNLM.png?branch=master)](https://travis-ci.org/linxihui/NNLM)
[![Coverage Status](http://codecov.io/github/linxihui/NNLM/coverage.svg?branch=master)](http://codecov.io/github/linxihui/NNLM?branch=master)

A package for Non-Negative Linear Models, including a fast non-negative least square (NNLS) solver and non-negative matrix factorization (NMF)

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
7. code coverage: trouble to get a badget 
8. ~~Parallel, openMP support~~
9. Support for missing values in NMF (can be used for imputation)

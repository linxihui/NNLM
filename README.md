# NNLM

[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/NNLM)](http://cran.r-project.org/package=NNLM)
[![Build Status](https://api.travis-ci.org/linxihui/NNLM.png?branch=master)](https://travis-ci.org/linxihui/NNLM)
[![Coverage Status](http://codecov.io/github/linxihui/NNLM/coverage.svg?branch=master)](http://codecov.io/github/linxihui/NNLM?branch=master)

This is a package for Nonnegative Linear Models (NNLM). It implements fast sequential coordinate descent algorithms for nonnegative linear regression and nonnegative matrix factorization (NMF or NNMF). It supports mean square error and Kullback-Leibler divergence loss. Many other features are also implemented, including missing value imputation, domain knowledge integration, designable W and H matrices and multiple forms of regularizations.

# Install

```r
# get a release version from CRAN
install.packages('NNLM')

# or a dev-version
library(devtools)
install_github('linxihui/NNLM')
```

# Why another NMF package?

A short answer: existent packages are not efficient for relative large matrices and lack many cool and new features.

# Features in the package?

1. Pattern extraction, as you would expect from an NMF package.
2. Multiple types of regularizations.
3. Designable matrix factorization to integrate domain/prior knowledge which is useful to many applications in bioinformatics, such as tumour content
deconvolution, pathway or subnetwork guided NMF for more biological meaningful decomposition, etc.
4. Built-in automatic missing value handling, which can then be used for efficient missing value imputation.
5. Utilize the power of imputation to create a _goodness-of-fit_ criterion which can be used to assess NMF performance and tune the rank `k`. 
(Not a built-in function yet and will be in next version shortly, but one can do it easily).
6. Parallel for one single NMF through openMP.

For more details, one can read the [vignette](https://cran.r-project.org/web/packages/NNLM/vignettes/Fast-And-Versatile-NMF.pdf) on CRAN.

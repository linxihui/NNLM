# NNLM

[![Build Status](https://api.travis-ci.org/linxihui/NNLM.png?branch=master)](https://travis-ci.org/linxihui/NNLM)
[![Coverage Status](http://codecov.io/github/linxihui/NNLM/coverage.svg?branch=master)](http://codecov.io/github/linxihui/NNLM?branch=master)

A package for Non-Negative Linear Models, including a fast non-negative least square (NNLS) solver and non-negative matrix factorization (NMF)

# Install

```r
library(devtools)
install_github('linxihui/NNLM')
```

# TODO
1. Add support for meta-genes: thresholding
2. Heatmap
3. ~~Examples~~
4. Vignette
5. ~~Test~~
6. ~~.traivs.yml~~
7. code coverage: trouble to get a badget 
8. ~~Parallel, openMP support~~
9. Support for missing values in NMF (how to handly in NNLS?) -> extremely usefull. Use NMF to impute missing values! (SVD for imputation as well)

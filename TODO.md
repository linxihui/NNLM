### Other thoughts on bioinformatics applications (DONE)

When the microarray contents multiple tumour, and $H_0$ is set to a 0/1 matrix represent tumour types, with 
number of rows equals to number of tumour, $W_1$ can be intepretted as tmuour type specific common profiles,
which can be used to characterise and distinguish tumours.


If $W$ is designed according to some known gene subnetworks, i.e., each column of $W$ represents a gene subnetwork,
by assigning 0 to genes not in the subnetwork and an unknown (and to be determined from $A$) non-negative number 
to genes within the subnetwork. One can solve the optimization problem to get the expression amount of each gene
within the subnetwork.  A gene can appear in different functional subnetwork, and the amonts in those different
subnetwork represent the different functionalities of the gene. (partial information)

-- to be implemented

$$ A = WH + W0 H1 + W1 H0 + W2 H3 + W3 H2 $$

$W, H$ are completely unknow
$W0, H0$ are completely known
$W2, H2$ are partially known (a mask, 0 or unkonw to be determined)

A careful design of $W0, H0, W2, H2$ may give disired result.

Systematic bias, or batch effect can be adjust by adding a column of 1 to $W_0$, and the correspondent coefficent
vector will capture the biasness.  In addtion, NMF has a natural property to elimate noise,
which is not necessary to be Gaussian noise.


## number of iteration can stop criterion of NNLS within NMF (DONE)
One don't need a very precise NNLS result in the early NMF iteration


## better regularization (DONE)
The current regularization has a weird behavior: RMSE converged but the targetted-error, which is
penalized-RMSE, keeps minizing the penalized-RMSE, probably by simply rescalling/distributing $W$ and $H$ matrix.
$W$ is intialized to have unit normal on each column, is there way to determine the optimal scale (minizing
the penalized-RMSE) of $W$ and $H$ assuming that RMSE is minized? 

$$A = W PDP^T H^T$$

where $W$ and $H$ has unit column norm. and $P$ is a permation matrix and $D$ is a diagonal matrix (scaling).
Can we sort columns of $W$ via magnitude of $D$?

## Can the sequential NNLS be extented to KL divergence? (DONE)


## Add support to user-specified inintialized $W$ matrix. (DONE)

## A = WH with nonnegative W, and unconstraint. Or both unconstraint

## Support for sparse matrix A

## Plotting

## Automatic Rank Tuning

## Probabilit like: p(a) = $\sum_{a=0} = p(a|b) p(b)}$$

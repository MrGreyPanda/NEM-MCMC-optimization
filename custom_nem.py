import nem
import numpy as np

def get_permissible_parents(n, permy, allparentRs):
    parentRs = np.empty(n, dtype=object)
    parentWeights = np.empty(n, dtype=object)
    parents = np.empty(n, dtype=object)
    nparents = np.empty(n, dtype=int)
    
    for i in range(n):
        parents[i] = np.where(permy == i)
        nparents[i] = len(parents[i])
        parentWeights[i] = [0.5] * nparents[i]
        if nparents[i] > 0:
            parentRs[i] = allparentRs[i][parents[i], :]
    
    return parents, nparents, parentWeights, parentRs


#' Permissible Parent Sets
#' 
#' helper function.
#' subset score tables by extracing/keeping only the rows of permissible parents (ie. parents further down in the permutation order)
#' @param n int,  n = #S-genes
#' @param permy int vectors, permutation order
#' @param allparentRs list of #S-genes elements. Each element is a data.frame containing the score tables for the corresponding S-gene, S-genes always in same order i.e rows are always S1, S2, etc.
#' @return parents: every list element contains the indices of permissible parents. 
#' nparents[i]= #parents S-gene i has. 
#' parentWeights: every list element contains a weight vector init with 0.5. 
#' parentRs: every list element contains a data.frame of dim #permissible parents x #E-genes (i.e. the subsetted score table)
#' @keywords permissible parents 
#' @export
#' @examples
#' get_permissible_parents()
# get_permissible_parents <- function(n, permy, allparentRs) {
  
#   # init
#   parentRs <- vector("list", n)
#   parentWeights <- vector("list", n)
#   parents <- vector("list", n)
#   nparents <- rep(NA, n) 
  
#   # subset of compatible parent sets
#   # For each node, we extract the rows compatible with the order from the corresponding score table of this node
#   for (ii in 1:n) {
#     parents[[ii]] <- permy[-which(permy == ii):-1]
#     nparents[ii] <- length(parents[[ii]]) # how many permissible parents it could have
#     parentWeights[[ii]] <- rep(0.5, nparents[ii]) # starting 0.5 for each parent
#     if (nparents[ii] > 0) {
#       parentRs[[ii]] <- allparentRs[[ii]][parents[[ii]],  ,drop=FALSE] 
#     }
#   }
#   return(list(parents=parents,nparents=nparents, parentWeights=parentWeights, parentRs=parentRs))
# }
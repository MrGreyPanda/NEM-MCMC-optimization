import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import py2rpy

# Install qlcMatrix if not already installed
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
try:
    qlcMatrix = importr('qlcMatrix')
except:
    utils.install_packages('qlcMatrix')
    qlcMatrix = importr('qlcMatrix')

def r_code(nodeLRs, permy):
    nodeLRs = py2rpy(nodeLRs)
    r = robjects.r
    # R code as a multiline string
    r_code = '''
 
    # calculation of the ancestor matrix:
    # or transitive closure
    ancestor <- function(incidence){
    incidence1 <- incidence
    incidence2 <- incidence
    k <- 1
    while (k < nrow(incidence)) {
        incidence1 <- incidence1%*%incidence
        incidence2 <- incidence2 + incidence1
        k <- k + 1
    }
    incidence2[which(incidence2 > 0)] <- 1
    return(incidence2)
    }


    # needed for optimising the LL
    localLLold <- function (x, a, b) {
    -sum(log(a*x + b))
    }
    localLL <- function (x, c) {
    -sum(log(c*x + 1))
    }


    # compute LRs for the cells when averaged over all positions
    computeCellLRs <- function(nodeLRs, nparents, parentWeights, parentRs) {
    n <- length(nparents)
    cellLRs <- nodeLRs # start with node log offset
    for (ii in 1:n) {
        if (nparents[ii] > 1) {
        cellLRs[, ii] <- cellLRs[, ii] + colSums(log(1 - parentWeights[[ii]] + parentWeights[[ii]]*parentRs[[ii]]))
        }
        if (nparents[ii] == 1) {
        cellLRs[, ii] <- cellLRs[, ii] + log(1 - parentWeights[[ii]] + parentWeights[[ii]]*parentRs[[ii]])
        }
    }
    cellLRs
    }


    # score an order by trying to optimise parent sets
    # under the marginal score
    NEMorderScore <- function(permy, nodeLRs, allparentRs) {
    
    parentRs <- vector("list", n)
    parentWeights <- vector("list", n)
    parents <- vector("list", n)
    nparents <- rep(NA, n) 
    
    # subset of compatible parent sets
    for (ii in 1:n) {
        parents[[ii]] <- which(permy > permy[ii])
        nparents[ii] <- length(parents[[ii]])
        parentWeights[[ii]] <- rep(0.5, nparents[ii])
        if (nparents[ii] > 0) {
        parentRs[[ii]] <- allparentRs[[ii]][parents[[ii]], ]
        }
    }
    
    # work out likelihood components of each attachment
    # we average over the parent sets with each weighted
    # in the end the weights should go to 0 or 1
    # to find the best parent set for the order
    
    oldLL <- -Inf
    LLdiff <- Inf
    
    while(LLdiff > 0.1) {
        
        cellLRs <- computeCellLRs(nodeLRs, nparents, parentWeights, parentRs)
        
        # total log score of each cell (should remove max etc)
        cellSums <- log(rowSums(exp(cellLRs)))
        # weight for each attachment point
        cellWeights <- exp(cellLRs - cellSums)
        
        LL <- sum(cellSums)
        
        # optimise each weight given old values
        
        parentWeightsNew <- parentWeights
        
        for (ii in 1:n) {
        if (nparents[ii] > 0) {
            for (jj in 1:nparents[ii]) {
            if (nparents[ii] > 1) {
                localvec <- parentRs[[ii]][jj, ]
            } else {
                localvec <- parentRs[[ii]]
            }
            
            Avec <- (localvec - 1)*cellWeights[, ii]
            Bvec <- 1 - parentWeights[[ii]][jj]*Avec + parentWeights[[ii]][jj]*(localvec - 1)
            Cvec <- Avec/Bvec
            
            parentWeightsNew[[ii]][jj] <- optimize(localLL, c(0, 1), tol = 0.01, c = Cvec)$minimum
            }
        }
        }
        
        # update for next loop
        parentWeights <- parentWeightsNew
        
        LLdiff <- LL - oldLL
        oldLL <- LL
        
        #print(LL)
    }
    
    # turn parent sets into DAG
    DAG <- matrix(0, n, n)
    parentWeightsDAG <- parentWeights
    for (ii in 1:n) {
        if (nparents[ii] > 0) {
        parentWeightsDAG[[ii]] <- 1*(parentWeights[[ii]] > 0.5)
        DAG[parents[[ii]], ii] <- parentWeightsDAG[[ii]]
        }
    }
    # score it
    cellLRs <- computeCellLRs(nodeLRs, nparents, parentWeightsDAG, parentRs)
    DAGLL <- sum(log(rowSums(exp(cellLRs))))
    
    # transitively complete DAG
    NEM <- ancestor(DAG)
    # score it
    parentWeightsNEM <- parentWeights
    for (ii in 1:n) {
        if (nparents[ii] > 0) {
        parentWeightsNEM[[ii]] <- NEM[parents[[ii]], ii]
        }
    }
    #  and give that to the order score
    cellLRs <- computeCellLRs(nodeLRs, nparents, parentWeightsNEM, parentRs)
    NEMLL <- sum(log(rowSums(exp(cellLRs))))
    
    return(list(NEMLL = NEMLL, NEM = NEM, DAGLL = DAGLL, DAG = DAG))
    }


    # order scheme for NEMs
    NEMorderMCMC <- function(startorder, nodeLRs, allparentRs, iterations, stepsave, moveprobs = c(0.95, 0.05), gamma = 1, NEMorDAG = TRUE){
    
    currentpermy <- startorder #starting order represented as a permutation
    currentScoreAll <- NEMorderScore(currentpermy, nodeLRs, allparentRs)
    if (NEMorDAG) {
        currentDAG <- currentScoreAll$NEM
        currentScore <- currentScoreAll$NEMLL
    } else {
        currentDAG <- currentScoreAll$DAG
        currentScore <- currentScoreAll$DAGLL
    }
    bestScore <- currentScore
    bestDAG <- currentDAG
    
    L1 <- list() # stores the adjacency matrix of DAG/NEM from the order
    L2 <- list() # stores its log score
    L3 <- list() # stores the order as a permutation
    
    zlimit<- floor(iterations/stepsave) + 1 # number of outer iterations
    length(L1) <- zlimit
    length(L2) <- zlimit
    length(L3) <- zlimit
    
    L1[[1]] <- currentDAG # starting DAG/NEM adjacency matrix
    L2[[1]] <- currentScore # starting DAG/NEM score
    L3[[1]] <- currentpermy # starting order
    
    for (z in 2:zlimit) { # the MCMC chain loop with 'iteration' steps is in two parts
        for (count in 1:stepsave) { # since we only save the results to the lists each 'stepsave'
        
        chosenmove <- sample.int(2, 1, prob = moveprobs)
        
        proposedpermy<-currentpermy #sample a new order by swapping two elements
        switch(as.character(chosenmove),
                "1"={ # swap any two elements at random
                sampledelements <- sample.int(n, 2, replace = FALSE) # chosen at random
                },
                "2"={ # swap any adjacent elements
                k <- sample.int(n - 1, 1) # chose the smallest at random
                sampledelements <- c(k, k + 1)
                })
        proposedpermy[sampledelements] <- currentpermy[rev(sampledelements)] # proposed new order
        
        proposedScoreAll <- NEMorderScore(proposedpermy, nodeLRs, allparentRs)
        if (NEMorDAG) {
            proposedDAG <- proposedScoreAll$NEM
            proposedScore <- proposedScoreAll$NEMLL
        } else {
            proposedDAG <- proposedScoreAll$DAG
            proposedScore <- proposedScoreAll$DAGLL
        }
        
        scoreratio <- exp(gamma*(proposedScore - currentScore)) # acceptance probability
        
        if (runif(1) < scoreratio) { # Move accepted then set the current order and scores to the proposal
            currentpermy <- proposedpermy
            currentDAG <- proposedDAG
            currentScore <- proposedScore
            if (currentScore > bestScore) {
            bestScore <- currentScore
            bestDAG <- currentDAG
            }
        }
        }
        
        L1[[z]] <- currentDAG # store adjacency matrix of DAG/NEM each 'stepsave'
        L2[[z]] <- currentScore # and its log score
        L3[[z]] <- currentpermy # and store current order
    }
    return(list(DAGs = L1, scores = L2, orders = L3, bestDAG = bestDAG, bestScore = bestScore))
    }


    set.seed(101)

    # number of nodes
    n <- 10
    # number of attachments
    N <- 100

    # LR table for attaching at the node directly
    # last column is detached completely
    nodeLRs <- matrix(rnorm(N*(n + 1)), N, n + 1)

    # R table for different parent sets
    # we store the exponentiated version and transposed!
    allparentRs <- vector("list", n)

    for (ii in 1:n) {
        allparentRs[[ii]] <- exp(t(matrix(rnorm(N*n), N, n)))
    }

    # pick an order
    permy <- sample(n)

    #NEMorderScore(permy, nodeLRs, allparentRs)

    start_time = Sys.time()
    NEMorderrun1 <- NEMorderMCMC(permy, nodeLRs, allparentRs, 4e3, 10)
    end_time = Sys.time()

    end_time - start_time

    plot(unlist(NEMorderrun1$scores), col = "dodgerblue", pch = 19)

    NEMorderrun1$bestScore

    start_time = Sys.time()
    NEMorderrun2 <- NEMorderMCMC(sample(n), nodeLRs, allparentRs, 4e3, 10)
    end_time = Sys.time()

    end_time - start_time

    NEMorderrun2$bestScore
    '''
    r(r_code)
    bestScore = r('NEMorderrun2$bestScore')
    print(bestScore)
    
r_code()


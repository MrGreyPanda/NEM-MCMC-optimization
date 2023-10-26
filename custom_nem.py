import nem
import numpy as np
# import optimizer
from scipy.optimize import minimize
from utils import ancestor, compute_ll_ratios, compute_ll

class CustomNEMOptimizer():
    def __init__(self, nem):
        self.nem = nem
        self.nem_order_score()
        
        
        
    
    def get_permissible_parents(self, n, permutation_order, allparentRs):
        parentRs = np.empty(n, dtype=object)
        parent_weights = np.empty(n, dtype=object)
        parents = np.empty(n, dtype=object)
        nparents = np.empty(n, dtype=int)
        print(f"Permutation_order: {permutation_order}")
        for i in range(n):
            index = np.where(permutation_order == i)[0][0]
            parents[i] = permutation_order[index + 1:]            
            nparents[i] = len(parents[i])
            parent_weights[i] = [0.5] * nparents[i]
            print(f"Parents weights of {i}: {parent_weights[i]}")
            if nparents[i] > 0:
                parentRs[i] = allparentRs[i][parents[i], :]
        
        return parents, nparents, parent_weights, parentRs


    # nodeLRs = U
    # nparents = nparents
    # parentWeights = parentWeights
    # parentRs = list of reduced score tables (without curent nodes scores)
    # n = numS

    def get_optimal_weights(self):
        # work out likelihood components of each attachment(E-gene)
        # we average over the parent sets with each weighted
        # in the end the weights should go to 0 or 1
        # to find the best parent set for the order

        oldLL = -float('inf')
        LLdiff = float('inf')

        while(LLdiff > 0.1):

            # total log score of each cell (logadd)
            # for easier computation
            cellSums = compute_ll(compute_ll_ratios(
                self.nem.parent_lists,
                self.nem.U,
                self.nem.parent_weights,
                self.nem.reduced_score_tables))

            # weight for each attachment point
            cellWeights = np.exp(self.nem.cell_ratios.T - cellSums)

            # total log likelihood
            LL = np.sum(cellSums)

            parent_weights = self.nem.parent_weights
            # -> we are here
            # optimise each weight given old values
            # Redo this part without using the legacy R code
            parent_weigths_new = parent_weights.copy()
            n = self.nem.num_s
            for i in range(n):
                if self.nem.n_parents[i] > 0:
                    for j in range(self.nem.n_parents[i]):
                        localvec = np.exp(self.nem.reduced_score_tables[i][j, :])
                        A = (localvec - 1) * cellWeights[:, i]
                        B = 1 - parent_weights[i][j] * A + parent_weights[i][j] * (localvec - 1)
                        C = A / B
                        def localLL(x, c):
                            return -np.sum(np.log(1 - x * A + x * (localvec - 1)) * c)
                        res = minimize(localLL, x0=0.5, bounds=[(0, 1)], args=(C,), method='L-BFGS-B', options={'ftol': 0.01})
                        parent_weigths_new[i][j] = res.x

            # update for next loop
            parent_weights = parent_weigths_new

            LLdiff = LL - oldLL
            oldLL = LL

        return parent_weights


    #permy = knockdown experiments
    #nodeLRs = U
    #allparentRs = list of reduced score tables (without curent nodes scores)
    #opt_fun = optimizer function
    def nem_order_score(self):
    
        # number of S-genes
        n = self.nem.num_s

        # subset of compatible parent sets
        parents, nparents, parentWeights, parentRs =  self.get_permissible_parents(self.nem.num_s, self.nem.permutation_order, self.nem.score_table_list)

        # work out likelihood components of each attachment(E-gene)
        # we average over the parent sets with each weighted
        # in the end the weights should go to 0 or 1
        # to find the best parent set for the order
        parent_weights = self.get_optimal_weights()

        print(f"Parent weights: {parent_weights}")
        # turn parent sets into DAG # TODO check if this is done right... I think it is right, but check, maybe write a test function
        DAG = np.zeros((n,n))
        parent_weights_dag = parent_weights
        for i in range(n):
            if nparents[i] > 0:
                parent_weights_dag[:,i] = parent_weights[:,i] > 0.5
                DAG[parents[[i]], i] = parent_weights_dag[[i]]
                
                        
        dag_ratios = compute_ll_ratios(self.nem.parent_lists, self.nem.U, parent_weights_dag, self.nem.reduced_score_tables)
        DAGLL = compute_ll(dag_ratios)

        # transitively complete DAG
        NEM = ancestor(DAG)
        # score it
        parentWeightsNEM = parentWeights
        for i in range(n):
            if nparents[i] > 0:
                parentWeightsNEM[[i]] = NEM[parents[[i]], i]
        
        
        # #  and give that to the order score
        nem_ll_ratios = compute_ll_ratios(self.nem.parent_lists, self.nem.U, parentWeightsNEM, self.nem.reduced_score_tables)
        NEMLL = compute_ll(nem_ll_ratios)

        self.NEMLL, self.NEM, self.DAGLL, self.DAG = NEMLL, NEM, DAGLL, DAG

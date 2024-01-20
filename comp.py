from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.linalg import solve_triangular, inv
import utils
import torch
from torch import optim

import numpy as np
import matplotlib.pyplot as plt

def local_ll_sum_γ(γ, c):
    return (-np.sum(np.log(γ * c + 1.0)), -np.sum(c / (γ * c + 1.0)))
    

def local_ll_sum_β(β, c):
    res = -np.sum(np.log(np.exp(β) * c + 1.0))
    return res

def local_ll_sum_w(w, c):
    res = -np.sum(np.log(expit(w) * c + 1.0))
    return res

# def local_ll_sum_b_inv(b, c, weights, i, k):
#     weights[i][k] = b
#     B = inv(np.eye(self.num_s) - self.exp_parent_weights(weights))
#     B = B / (1.0 + B)
#     res = -np.sum(np.log(B[i][k] * c + 1.0))
#     return res

def iden(x):
    return x

class Comp:
    def __init__(self, order, num_s, num_e, U, score_tables):
        self.order = order
        self.num_s = num_s
        self.num_e = num_e
        self.U = U
        self.score_tables = score_tables
        self.I = np.eye(self.num_s)
        self.mask = np.zeros((self.num_s, self.num_s))
        self.mask = self.get_permissible_parents(order, self.mask, init_val=1.0, init=True)
        
        
    def get_permissible_parents(self, perm_order, weights, init_val=0.5, i1=None, i2=None, init=False):
        parents_list = np.empty(self.num_s, dtype=object)
        if not init:
            weights[i1] = 0
            weights[i2] = 0
            weights[:, i1] = 0
            weights[:, i2] = 0
        for i in range(self.num_s):
            index = np.where(perm_order == i)[0][0]
            parents_list[i] = perm_order[:index]
            if init:
                for j in parents_list[i]:
                    weights[i][j] = init_val
            else:
                if i1 in parents_list[i]:
                    weights[i][i1] = init_val
                elif i2 in parents_list[i]:
                    weights[i][i2] = init_val
                elif i == i1 or i == i2:
                    for j in parents_list[i]:
                        weights[j][i] = init_val
        self.parents_list = parents_list
        return weights
        
    def create_dag(self, weights):
        dag_weights = weights.copy()
        dag_weights = 1 * (weights > 0.5)
        return dag_weights

    def compute_cell_ratios(self, weights, score_tables):
        cell_ratios = self.U.copy()
        for i in range(self.num_s): 
            for j in self.parents_list[i]:
                cell_ratios[i, :] += np.log(1.0 -
                                            weights[i][j] +
                                            weights[i][j] *
                                            np.exp(score_tables[i][j]))
        return cell_ratios

    def calculate_ll(self, cell_ratios):
        cell_sums = np.logaddexp.reduce(cell_ratios, axis=0)
        order_weights = np.exp(cell_ratios - cell_sums)
        ll = np.sum(cell_sums)
        return order_weights, ll
    
    def expit_parent_weights(self, weights):
        new_weights = weights.copy()
        for i in range(self.num_s):
            for j in self.parents_list[i]:
                new_weights[i][j] = expit(new_weights[i][j])
        return new_weights
    
    def calculate_local_optimum_γ(self, i, k, order_weights, weights, bounds):
        local_vec = np.exp(self.score_tables[i][k])
        a = (local_vec - 1.0) * order_weights[i]
        b = 1.0 - weights[i][k] * a + weights[i][k] * (local_vec - 1.0)
        c = a / b
        res = minimize(local_ll_sum_γ, x0=weights[i][k], bounds=bounds, args=(c,), jac=True, method='L-BFGS-B', tol=0.01)
        if res.success is False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        return res.x 

    def opt_γ(self, weights, bounds):
        cell_ratios = self.compute_cell_ratios(weights, self.score_tables)
        order_weights, ll = self.calculate_ll(cell_ratios)
        new_parent_weights = weights.copy()
        for i in range(self.num_s):
            for k in self.parents_list[i]:
                new_parent_weights[i][k] = self.calculate_local_optimum_γ(i, k, order_weights, new_parent_weights, bounds)
        weights = new_parent_weights.copy()
        return ll, weights
    
    def calculate_local_optimum_β(self, i, k, order_weights, weights, bounds):
        local_vec = np.exp(self.score_tables[i][k])
        a = (local_vec - 1.0) * order_weights[k]
        b = 1.0 - np.exp(weights[i][k]) * a + np.exp(weights[i][k]) * (local_vec - 1.0)
        c = a / b
        res = minimize(local_ll_sum_β, x0=weights[i][k], bounds=bounds, args=(c,), method='L-BFGS-B', tol=0.01)
        if res.success is False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        return res.x

    def opt_β(self, weights, bounds):
        cell_ratios = self.compute_cell_ratios(np.exp(weights), self.score_tables)
        order_weights, ll = self.calculate_ll(cell_ratios)
        new_parent_weights = weights.copy()
        for i in range(self.num_s):
            for k in self.parents_list[i]:
                new_parent_weights[i][k] = self.calculate_local_optimum_β(i, k, order_weights, new_parent_weights, bounds)
        weights = new_parent_weights.copy()
        return ll, weights
        # B = weights.copy()
        # B = inv(np.eye(self.num_s) - B) - np.eye(self.num_s)
        # B = 1.0 * (B > 0.5)
        # _, real_ll = self.calculate_ll(self.compute_cell_ratios(B, self.score_tables))
        # print(f"Real LL: {real_ll}")
        # print(f"B sum: {np.sum(B)}")
        # return real_ll, B
    
    def calculate_local_optimum_w(self, i, k, order_weights, weights, bounds):
        local_vec = np.exp(self.score_tables[i][k])
        a = (local_vec - 1.0) * order_weights[k]
        b = 1.0 - expit(weights[i][k]) * a + expit(weights[i][k]) * (local_vec - 1.0)
        c = a / b
        res = minimize(local_ll_sum_w, x0=weights[i][k], bounds=bounds, args=(c,), method='L-BFGS-B', tol=0.01)
        if res.success is False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        
        return res.x
    
    def opt_w(self, weights, bounds):
        cell_ratios = self.compute_cell_ratios(self.expit_parent_weights(weights), self.score_tables)
        order_weights, ll = self.calculate_ll(cell_ratios)
        new_parent_weights = weights.copy()
        for i in range(self.num_s):
            for k in self.parents_list[i]:
                new_parent_weights[i][k] = self.calculate_local_optimum_w(i, k, order_weights, new_parent_weights, bounds)
                
        weights = new_parent_weights.copy()
            
        return ll, weights
    
    def log_parent_weights(self, weights):
        for i in range(self.num_s):
            for j in self.parents_list[i]:
                weights[i][j] = np.log(weights[i][j])
                
        return weights

    def exp_parent_weights(self, weights):
        new_weights = weights.copy()
        for i in range(self.num_s):
            for j in self.parents_list[i]:
                new_weights[i, j] = np.exp(new_weights[i, j])
                
        return new_weights
    
    def local_ll_sum_b_inv(self, x, weights, i, k, local_vec, a_vec):
        weights[i][k] = x
        B = inv(self.I - self.exp_parent_weights(weights))
        B = B / (1.0 + B)
        b_vec = 1.0 - B[i][k] * a_vec + B[i][k] * (local_vec - 1.0)
        c_vec = a_vec / b_vec
        res = -np.sum(np.log(B[i][k] * c_vec + 1.0))
        return res
    
    def calculate_local_optimum_b_inv(self, i, k, order_weights, weights):
        local_vec = np.exp(self.score_tables[i][k])
        a_vec = (local_vec - 1.0) * order_weights[k]
        # b_vec = 1.0 - expit_weights[i][k] * a_vec + expit_weights[i][k] * (local_vec - 1.0)
        # c_vec = a_vec / b_vec
        res = minimize(self.local_ll_sum_b_inv, x0=weights[i][k], bounds=[(-1000, 1000)], options={'eps': 1e-6}, args=(weights, i, k, local_vec, a_vec), method='L-BFGS-B', tol=0.001)
        if res.success is False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        
        return res.x
    
    def opt_b(self, weights):
        new_parent_weights = weights.copy()
        inv_weights = inv(self.I - self.exp_parent_weights(new_parent_weights))
        # expit_weights = inv_weights / (1.0 + inv_weights)
        cell_ratios = self.compute_cell_ratios(inv_weights / (1.0 + inv_weights), self.score_tables)
        order_weights, ll = self.calculate_ll(cell_ratios)
        for i in range(self.num_s):
            for k in self.parents_list[i]:
                new_parent_weights[i][k] = self.calculate_local_optimum_b_inv(i, k, order_weights, new_parent_weights)
        weights = new_parent_weights.copy()
        return ll, weights
    
    def optimize(self):
        bounds = [(-1000, 1000)]
        weights = np.zeros((self.num_s, self.num_s))
        weights = self.get_permissible_parents(perm_order=self.order, weights=weights, init=True, init_val=-0.6931471806)
        max_iter = 30
        ll_diff = float('inf')
        ll_old = -float('inf')
        ll_list = []
        weight_list = []
        best_ll = -float('inf')
        best_index = 0
        iter_count = 0
        while iter_count < max_iter and ll_diff > 0.0001:
            ll, weights = self.opt_b(weights)
            ll_list.append(ll)
            if ll > best_ll:
                best_ll = ll
                best_index = iter_count
            weight_list.append(weights)
            ll_diff = np.abs(ll - ll_old)
            ll_old = ll
            iter_count += 1
            print(f"LL: {ll}, ll_diff: {ll_diff}")
        
        weights = weight_list[best_index]
        B_tilde = inv(self.I - np.exp(weights)) - np.eye(self.num_s)
        B_tilde = B_tilde / (1.0 + B_tilde)
        B_tilde = 1 * (B_tilde > 0.5)
        _, real_ll = self.calculate_ll(self.compute_cell_ratios(B_tilde, self.score_tables))
        print(f"Real LL: {real_ll}")
        return B_tilde.T, real_ll
    
    
    def optimize_weights_with_expit(self, weights, bounds):
        i = 0
        max_iter = 1000
        ll_diff = float('inf')
        ll_old = -float('inf')
        while i < max_iter and ll_diff > 0.1:
            cell_ratios = self.compute_cell_ratios(self.expit_parent_weights(weights), self.score_tables)
            order_weights, ll = self.calculate_ll(cell_ratios)
            new_parent_weights = weights.copy()
            for i in range(self.num_s):
                for k in self.parents_list[i]:
                    new_parent_weights[i][k] = self.calculate_local_optimum_w(i, k, order_weights, new_parent_weights, bounds)
            weights = new_parent_weights.copy()
            ll_diff = np.abs(ll - ll_old)
            ll_old = ll
            i += 1
            print(f"LL: {ll}, ll_diff: {ll_diff}")
        return ll, weights 
    
    def optimize_weights_normal(self, weights, bounds):
        weights_list = []
        ll_list = []
        i = 0
        max_iter = 1000
        ll_diff = float('inf')
        ll_old = -float('inf')
        while i < max_iter and ll_diff > 0.1:
            cell_ratios = self.compute_cell_ratios(weights, self.score_tables)
            order_weights, ll = self.calculate_ll(cell_ratios)
            new_parent_weights = weights.copy()
            for i in range(self.num_s):
                for k in self.parents_list[i]:
                    new_parent_weights[i][k] = self.calculate_local_optimum_γ(i, k, order_weights, new_parent_weights, bounds)
            weights = new_parent_weights.copy()
            ll_diff = ll - ll_old
            ll_old = ll
            weights_list.append(weights)
            ll_list.append(ll)
            i += 1
            print(f"LL: {ll}, ll_diff: {ll_diff}")
        return ll, weights, weights_list, ll_list
    
    def compare(self, n_iters):
        # scr_lst_γ, scr_lst_β, scr_lst_w, scr_lst = [], [], [], []
        # best_scrs_γ, best_scrs_β, best_scrs_w = [], [], [], []
        γ, β, w = np.zeros((self.num_s, self.num_s)), np.zeros((self.num_s, self.num_s)), np.zeros((self.num_s, self.num_s))
        
        γ = self.get_permissible_parents(perm_order=self.order, weights=γ, init=True, init_val=0.5)
        β = self.get_permissible_parents(perm_order=self.order, weights=β, init=True, init_val=-0.0)
        w = self.get_permissible_parents(perm_order=self.order, weights=w, init=True, init_val=0.0)
    
        # opt_method_lst = [self.opt_γ, self.opt_β, self.opt_w]
        names = ["γ", "β", "w", "β_inv"]
        n = len(names)
        bounds_γ = [(0.0, 1.0)]
        bounds_β = [(-5.0, 0.0)]
        bounds_w = [(-float('inf'), float('inf'))]
        weight_lst = [γ, β, w, β.copy()]
        scr_lsts_lst = [[] for _ in range(n)]
        best_scr_lst = [[] for _ in range(n)]
        bounds_lst = [bounds_γ, bounds_β, bounds_w, bounds_β.copy()]
        # γ_lst, β_lst, w_lst, β_inv_lst = [], [], [], []
        weights_lst = [[] for _ in range(n)]
        # ll_sum_fun = [local_ll_sum_gamma, local_ll_sum_beta, local_ll_sum_w]
        method_lst = [self.opt_γ, self.opt_β, self.opt_w, self.opt_b]
        
        iter_count = 0
        while not(iter_count > 10000):
            for j in range(n):
                print(f"Optimizing {names[j]}...")
                scr, curr_weights = method_lst[j](weight_lst[j], bounds_lst[j])
                weights_lst[j].append(curr_weights)
                scr_lsts_lst[j].append(scr)
                best_scr_lst[j].append(max(scr_lsts_lst[j]))
                print(f"Score for {names[j]}: {scr}")
            # print(f"Score Diffs, Beta_inv - Beta: {scr_lsts_lst[3][-1] - scr_lsts_lst[1][-1]}")
            # print(f"Score Diffs: {scr_lst_γ[-1] - scr_lst_β[-1]}, {scr_lst_γ[-1] - scr_lst_w[-1]}, {scr_lst_β[-1] - scr_lst_w[-1]}")
            print("---------------------------------")
            iter_count += 1
            
        
    # def opt_with_torch(self):
    #     weights = np.zeros((self.num_s, self.num_s))
    #     weights = self.get_permissible_parents(perm_order=self.order, weights=weights, init=True, init_val=-0.6931471806)
    #     weights = torch.tensor(weights, requires_grad=True)
    #     weight_list = []
    #     inv_weights = torch.inverse(torch.eye(self.num_s) - self.exp_parent_weights(weights))
    #     expit_weights = inv_weights / (1.0 + inv_weights)
    #     order_weights, ll = self.calculate_ll(self.compute_cell_ratios(expit_weights, self.score_tables))
    #     weight_list.append(weights)
    #     best_ll = -float('inf')
    #     best_weights = weights
    #     max_iter = 100
    #     bounds = [(0.0, 1.0)] * self.num_s * self.num_s
    #     # optimizer = optim.Adam([weights], lr=0.01, betas=(0.9, 0.999), eps=1e-08)
    #     optimizer = optim.SGD([weights], lr=0.01, momentum=0.9, nesterov=True)
    #     mask = torch.zeros((self.num_s, self.num_s))
    #     for i in range(self.num_s):
    #         for j in self.parents_list[i]:
    #             mask[i][j] = 1
    #     for i in range(max_iter):
    #         print(f"Iteration: {i}")
    #         with torch.enable_grad():
    #             loss = self.loss_fun(weights, order_weights)
    #             loss.backward()
    #             weights.grad *= mask
    #             optimizer.step()
    #             weight_list.append(weights.detach().numpy())
    #         with torch.no_grad():
    #             # for ind in range(self.num_s):
    #             #     for j in range(self.num_s):
    #             #         if j not in self.parents_list[ind]:
    #             #             weights[ind][j] = 0
                    
    #             inv_weights = torch.inverse(torch.eye(self.num_s) - self.exp_parent_weights(weights))
    #             expit_weights = inv_weights / (1.0 + inv_weights)
    #             mini = float('inf')
    #             maxi = -float('inf')
    #             for ind in range(self.num_s):
    #                 for jnd in self.parents_list[ind]:
    #                     if weights[ind][jnd] > maxi:
    #                         maxi = inv_weights[ind][jnd]
    #                     if weights[ind][jnd] < mini:
    #                         mini = inv_weights[ind][jnd]
    #             print(f"Max: {maxi}, Min: {mini}")
    #             order_weights, ll = self.calculate_ll(self.compute_cell_ratios(expit_weights, self.score_tables))
    #             print(f"LL: {ll}")
    #             if ll > best_ll:
    #                 best_ll = ll
    #                 best_weights = weights.detach().numpy()
    #     inv_weights = inv(np.eye(self.num_s) - self.exp_parent_weights(weights))
    #     expit_weights = inv_weights / (1.0 + inv_weights)
    #     B = 1.0 * (expit_weights > 0.5)
    #     return B.T, best_ll
    
    # def loss_fun(self, weights, order_weights):
    #     res = 0
    #     for i in range(self.num_s):
    #         for k in self.parents_list[i]:
    #             local_vec = torch.exp(torch.tensor(self.score_tables[i][k]))
    #             a_vec = (local_vec - 1.0) * order_weights[k]
    #             B = torch.inverse(torch.eye(self.num_s) - self.exp_parent_weights(weights))
    #             B = B / (1.0 + B)
    #             b_vec = 1.0 - B[i][k] * a_vec + B[i][k] * (local_vec - 1.0)
    #             c_vec = a_vec / b_vec
    #             res += -torch.sum(torch.log(B[i][k] * c_vec + 1.0))
    #     return res
                
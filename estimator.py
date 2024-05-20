import pandas as pd
import numpy as np
import random
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def solve_OLS(O, Z, tau_init=0):
    '''
    Regress O on Z_1, Z_2, ...; the regression coefficient is tau_1, tau_2, ...
    Using standard linear regression to solve the OLS problem.

    Parameters:
    -------------
    O: the matrix to be regressed
    Z: a list of intervention matrices
    tau_init: unused

    Returns:
    -----------
    tau: the vector of regression coefficients
    '''

    m, n = O.shape
    y = O.reshape(-1)
    X = np.hstack([Z_i.reshape(-1, 1) for Z_i in Z])

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    tau = reg.coef_

    return tau


class Node:

    def __init__(self, estimator, treatment_name, node_name):
        self.estimator = estimator # pointer to the main estimator instance
        self.treatment_name = treatment_name
        self.node_name = node_name
        self.left = None
        self.right = None
        self.split, self.value = None, None
        
        self.columns_for_X = self.estimator.columns_for_X
        self.columns_for_Z = self.estimator.columns_for_Z
        self.column_unit = self.estimator.column_unit
        self.column_time = self.estimator.column_time
        self.min_samples_leaf = self.estimator.min_samples_leaf

        self.cluster_mask = self.estimator.data[f'Cluster_{self.treatment_name}'] == self.node_name
        self.n = self.cluster_mask.sum()
        self.tau = None

    def get_split(self): # binary search for split
        self.data_in_leaf = self.estimator.data[self.cluster_mask]

        # make the target for this node: (O-M-sum_{all but current treatment} tau*Z)
        excluded_column_name = f"{self.treatment_name}_{self.node_name}"        
        filtered_Z_df = self.estimator.Z_list_clusters_df.drop(columns=[excluded_column_name])
        filtered_tau = np.delete(self.estimator.tau, self.estimator.Z_list_clusters_df.columns.get_loc(excluded_column_name))
        self.target = self.estimator.data['target'] - filtered_Z_df.dot(filtered_tau)

        best_split, best_value = None, -np.inf
        selected_columns = random.sample(self.columns_for_X, len(self.columns_for_X) // 2) if self.estimator.random else self.columns_for_X
        for column in selected_columns:
            unique_values = sorted(self.data_in_leaf[column].unique())

            # Binary search implementation
            low, high = 0, len(unique_values) - 1
            while low <= high:
                mid = low + (high - low) // 2
                split_value = unique_values[mid]

                left_leaf = self.cluster_mask & (self.estimator.data[column] <= split_value) # indicators for left leaf
                right_leaf = self.cluster_mask & (self.estimator.data[column] > split_value)
                value = self.calculate_value(left_leaf, right_leaf, column, split_value)

                if value > best_value and not self.is_invalid_split(left_leaf, right_leaf): 
                    best_split = (column, split_value)
                    best_value = value

                # Adjust search space
                if 0 < mid < len(unique_values) - 1:
                    prev_value = self.calculate_value(
                        self.cluster_mask & (self.estimator.data[column] <= unique_values[mid-1]), 
                        self.cluster_mask & (self.estimator.data[column] > unique_values[mid-1]))
                    next_value = self.calculate_value(
                        self.cluster_mask & (self.estimator.data[column] <= unique_values[mid+1]), 
                        self.cluster_mask & (self.estimator.data[column] > unique_values[mid+1]))
                    
                    if prev_value > next_value: high = mid - 1
                    else: low = mid + 1
                else: break
        self.split, self.value = best_split, best_value
        return self

    def is_invalid_split(self, left_leaf, right_leaf):
        # Check minimum sample size requirement
        if np.sum(left_leaf) < self.min_samples_leaf or np.sum(right_leaf) < self.min_samples_leaf:
            return True

        data_for_pivot = self.estimator.data[[self.column_unit, self.column_time]]
        left_leaf_pivoted = data_for_pivot.assign(Indicator=left_leaf).pivot_table(index=self.column_unit, columns=self.column_time, values='Indicator').values
        right_leaf_pivoted = data_for_pivot.assign(Indicator=right_leaf).pivot_table(index=self.column_unit, columns=self.column_time, values='Indicator').values
        Z = self.estimator.Z_list[self.estimator.columns_for_Z.index(self.treatment_name)]

        Z_left = Z * left_leaf_pivoted
        Z_right = Z * right_leaf_pivoted

        # Check if the resulting arrays are all zeros or already in Z_list_clusters
        if not Z_left.any() or not Z_right.any() or \
            any((Z_left == Z_cluster).all() for Z_cluster in self.estimator.Z_list_clusters) or \
            any((Z_right == Z_cluster).all() for Z_cluster in self.estimator.Z_list_clusters):
            return True

        # If none of the conditions above are met, it's a valid split
        return False

    def calculate_value(self, left_leaf, right_leaf, column=None, split_value=None, verbose=False):
        Z_values = self.estimator.data[self.treatment_name] 

        left_treated_target = self.target[Z_values & left_leaf]
        right_treated_target = self.target[Z_values & right_leaf]
        treated_target = self.target[Z_values & self.cluster_mask]
        self.tau, tau_left, tau_right =  treated_target.mean(), left_treated_target.mean(), right_treated_target.mean()

        if len(left_treated_target) <= 5 or len(right_treated_target) <= 5: return -np.inf

        if self.estimator.splitting_criterion == "MAE":
            ret = np.sum(np.abs(self.tau - treated_target))
            ret -= np.sum(np.abs(tau_left - left_treated_target))
            ret -= np.sum(np.abs(tau_right - right_treated_target))
        elif self.estimator.splitting_criterion == "MSE":
            ret = np.sum(np.abs(self.tau - treated_target)**2)
            ret -= np.sum(np.abs(tau_left - left_treated_target)**2)
            ret -= np.sum(np.abs(tau_right - right_treated_target)**2)
        
        t_stat, p_val = ttest_ind(left_treated_target, right_treated_target, equal_var=False)

        if p_val > self.estimator.p_value_for_splits or ret <= 0: return -np.inf
        return ret     


class TreatmentEffectEstimator:

    def __init__(self, data, column_unit, column_time, column_outcome, columns_for_X, columns_for_Z, suggest_r, 
                 min_samples_leaf=10, 
                 p_value_for_splits=.02, 
                 random=False,
                 force_center_split=False,
                 splitting_criterion="MSE"):
        self.data = data
        self.column_unit = column_unit
        self.column_time = column_time
        self.column_outcome = column_outcome
        self.columns_for_X = columns_for_X  + [column_outcome]
        self.columns_for_Z = columns_for_Z
        self.suggest_r = suggest_r
        self.min_samples_leaf = min_samples_leaf
        self.p_value_for_splits = p_value_for_splits
        self.random = random
        self.force_center_split = force_center_split
        self.splitting_criterion = splitting_criterion

        self.O = data.pivot_table(index=self.column_unit, columns=self.column_time, values=column_outcome)
        n, T = self.O.shape

        self.l = np.linalg.svd(self.O, compute_uv=False)[0]
        self.Z_list = [data.pivot_table(index=self.column_unit, columns=self.column_time, values=column).values for column in columns_for_Z]
        self.M = np.copy(self.O)
        self.m = np.zeros(n)

        for treatment_name in columns_for_Z: self.data[f'Cluster_{treatment_name}'] = 0 # cluster labels for each treatment
        self.roots = {treatment_name: Node(self, treatment_name, 0) for treatment_name in columns_for_Z}
        self.leaves = {treatment_name: [self.roots[treatment_name]] for treatment_name in self.roots}

    def update_M_m(self, eps=1e-3):
        O, Z_list, M, m, l = self.O.values, self.Z_list, self.M, self.m, self.l

        Z_list, self.clusters, self.treatment_names, self.cluster_names = [], [], [], []
        for treatment_name, Z in zip(self.columns_for_Z, self.Z_list):
            clusters = self.data.pivot_table(index=self.column_unit, columns=self.column_time, values=f'Cluster_{treatment_name}')
            cluster_names_unique = pd.unique(clusters.values.ravel())
            
            matrices_for_treatment = [np.where(clusters == element, 1, 0) for element in cluster_names_unique]
            self.clusters.extend(matrices_for_treatment)

            self.treatment_names.extend([treatment_name] * len(matrices_for_treatment))
            self.cluster_names.extend(cluster_names_unique)

            Z_list.extend([Z * matrix for matrix in matrices_for_treatment])
        self.Z_list_clusters = Z_list
        
        self.Z_list_clusters_df = self.data[[self.column_unit, self.column_time]].copy()
        for Z, treatment_name, cluster_name in zip(self.Z_list_clusters, self.treatment_names, self.cluster_names):
            df = pd.DataFrame(Z, index=self.O.index, columns=self.O.columns)
            df = df.reset_index().melt(id_vars=[self.column_unit], var_name=self.column_time, value_name=f'{treatment_name}_{int(cluster_name)}')
            self.Z_list_clusters_df = self.Z_list_clusters_df.merge(df, on=[self.column_unit, self.column_time], how='left')
        self.Z_list_clusters_df.drop(columns=[self.column_unit, self.column_time], inplace=True)

        self.norms = [np.linalg.norm(Z, 'fro') for Z in Z_list]
        Z_list = [Z/norm for Z, norm in zip(Z_list, self.norms)]

        n, T = O.shape
        tau = np.zeros(len(Z_list))
        for _ in range(400):
            target = O - M - np.outer(m, np.ones(T))
            tau_new = solve_OLS(target, Z_list)

            M = O - sum(tau_new[k] * Z for k, Z in enumerate(Z_list)) - np.outer(m, np.ones(T))
            u, s, vh = np.linalg.svd(M, full_matrices=False)
            s = np.maximum(s - l, 0)
            M = (u * s).dot(vh)

            m = (O - M - sum(tau_new[k] * Z for k, Z in enumerate(Z_list))).dot(np.ones((T, 1))) / T 

            if np.sum(s > 0) < self.suggest_r: 
                l *= 0.9  
                continue

            if np.linalg.norm(tau_new - tau) < eps * np.linalg.norm(tau): break
            tau = tau_new.copy()
            
        target = pd.DataFrame(target, index=self.O.index, columns=self.O.columns).reset_index().melt(id_vars=[self.column_unit], var_name=self.column_time, value_name='target')
        if 'target' in self.data.columns: self.data.drop(columns='target', inplace=True)
        self.data = pd.merge(self.data, target, on=[self.column_unit, self.column_time], how='left')

        self.M, self.m, self.l = M, m, l
        self.tau = [t/n for t, n in zip(tau_new, self.norms)]
        return tau_new 

    def update_tree(self):
        if all(leaf.value == -np.inf for leaves in self.leaves.values() for leaf in leaves): 
            return -np.inf

        self.update_M_m()

        for treatment_name, leaves in self.leaves.items():
            [leaf.get_split() for leaf in leaves if leaf.value != -np.inf]

        # Iterate and split for each treatment
        for treatment_name, leaves in self.leaves.items():
            # Find the leaf with the max value
            leaf = max(leaves, key=lambda x: x.value)

            if self.force_center_split: ## newly added
                leaf.split = self.columns_for_X[0], self.data[self.columns_for_X[0]].median()
                leaf.value = 0

            if leaf.value == -np.inf: continue
            split_column, split_value = leaf.split

            # print(treatment_name, leaf.split)
            # print([(le.node_name, le.value, le.split) for le in leaves])
            # print(sorted([(le.tau, le.n) for le in leaves], key=lambda x: x[0]))

            # Perform the split and update data clusters
            cluster_column_name = f'Cluster_{treatment_name}'
            max_cluster = max(self.data[cluster_column_name])
            left_cluster, right_cluster = max_cluster + 1, max_cluster + 2
            cluster_mask = self.data[cluster_column_name] == leaf.node_name
            self.data.loc[cluster_mask & (self.data[split_column] <= split_value), cluster_column_name] = left_cluster
            self.data.loc[cluster_mask & (self.data[split_column] > split_value), cluster_column_name] = right_cluster

            # Update tree structure
            leaf.left, leaf.right = Node(self, treatment_name, left_cluster), Node(self, treatment_name, right_cluster)
            leaves.remove(leaf)
            leaves.extend([leaf.left, leaf.right])


    def debias(self):
        tau = self.update_M_m(eps=1e-13)
        O, M = self.O.values, self.M
        Z_list = [Z/norm for Z, norm in zip(self.Z_list_clusters, self.norms)] 

        u, s, vh = np.linalg.svd(M, full_matrices=False)
        r = np.sum(s / np.cumsum(s) >= 1e-6) 
        u = u[:, :r]
        vh = vh[:r, :]

        n, T = u.shape[0], vh.shape[1]
        I_VVT = np.eye(T) - vh.T.dot(vh)
        one_T = np.ones((T, 1))

        PTperpZ_list = []
        for Z in Z_list:
            PTperpZ = (np.eye(n) - u.dot(u.T)).dot(Z).dot(np.eye(T) - one_T @ one_T.T / T).dot(I_VVT)
            PTperpZ_list.append(PTperpZ)

        D = np.array([[np.sum(Zk * PTperpZ) for PTperpZ in PTperpZ_list] for Zk in Z_list])
        Delta = self.l * np.array([np.sum(Z * (u.dot(vh) - u.dot(vh) @ one_T @ one_T.T / T)) for Z in Z_list])

        tau_delta = np.linalg.pinv(D) @ Delta
        tau_debias = tau - tau_delta
        tau_debias = [t/n for t, n in zip(tau_debias, self.norms)]
        self.tau = tau_debias

        # write to dataframe
        results = {treatment_name: np.zeros_like(O, dtype=float) for treatment_name in self.columns_for_Z}
        for treatment_name, c, t in zip(self.treatment_names, self.clusters, tau_debias): 
            results[treatment_name] += c * t

        self.results_df = self.data[[self.column_unit, self.column_time]]
        for treatment_name in results:
            df = pd.DataFrame(results[treatment_name], index=self.O.index, columns=self.O.columns).reset_index().melt(id_vars=self.column_unit, var_name=self.column_time, value_name=treatment_name + '_result')
            self.results_df = pd.merge(self.results_df, df, on=[self.column_unit, self.column_time], how='left')


    def fit(self, max_leaves = 20):
        for i in range(max_leaves): self.update_tree()
        self.debias()

    def effect(self):
        return self.results_df.drop(columns=[self.column_unit, self.column_time])
    

    
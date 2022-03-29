###############################################################################
# Very simple python-native implementation of CART for continuous outcomes 
# and continuous predictors
# 
# NOTE: This largely tracks the R implementation, but in the future will be 
# refactored to a simpler and more pythonic implementation
###############################################################################

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import deque


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.expected_value = None
        self.split_variable = None
        self.split_point = None
        self.parent = None
        self.node_origin_type = 0
        self.node_type = 1
    
    def subset_data(self, x):
        n, p = x.shape
        subset_cond = np.repeat(True, n)
        node_has_parent = self.parent is not None
        current_node = self
        while node_has_parent:
            parent_node_cutoff = current_node.parent.split_point
            parent_node_split_var = current_node.parent.split_variable
            current_node_type = current_node.node_origin_type
            if current_node_type == 1:
                current_node_cond = x[:,parent_node_split_var] > parent_node_cutoff
            elif current_node_type == -1:
                current_node_cond = x[:,parent_node_split_var] <= parent_node_cutoff
            else:
                current_node_cond = subset_cond
            subset_cond = subset_cond * current_node_cond
            current_node = current_node.parent
            node_has_parent = current_node.parent is not None
        return subset_cond


def split_sum_of_squares(cutoff, x, y):
    # Original sum of squares
    y_bar_t = np.mean(y)
    ss_t = np.sum(np.power(y-y_bar_t, 2))

    # New sum of squares
    left_subset = x <= cutoff
    right_subset = x > cutoff
    y_bar_l = np.mean(y[left_subset])
    ss_l = np.sum(np.power(y[left_subset] - y_bar_l, 2))
    y_bar_r = np.mean(y[right_subset])
    ss_r = np.sum(np.power(y[right_subset] - y_bar_r, 2))

    return ss_t - ss_l - ss_r

def best_split(X, y, i):
    min_result = minimize(lambda x: -split_sum_of_squares(x, X[:,i], y), np.mean(X[:,i]))
    return (min_result.x[0], -min_result.fun)

def num_leaves(root_node):
    visit_nodes = deque()
    current_node = root_node
    visit_nodes.append(current_node.left)
    visit_nodes.append(current_node.right)
    MAX_ITER = 10000
    iter_count = 0

    answer = 0
    while len(visit_nodes) > 0 and iter_count < MAX_ITER:
        current_node = visit_nodes.popleft()
        if current_node.node_type == 1:
            answer += 1
        else:
            visit_nodes.append(current_node.left)
            visit_nodes.append(current_node.right)
    return answer

def leaf_extract(X, root_node):
    n, p = X.shape
    leaf_count = num_leaves(root_node)
    leaf_vector = np.empty(leaf_count)
    leaf_subsets = np.empty((n, leaf_count))

    visit_nodes = deque()
    current_node = root_node
    visit_nodes.append(current_node.left)
    visit_nodes.append(current_node.right)
    MAX_ITER = 10000
    iter_count = 0
    leaf_counter = 0

    while len(visit_nodes) > 0 and iter_count < MAX_ITER:
        current_node = visit_nodes.popleft()
        if current_node.node_type == 1:
            leaf_subsets[:,leaf_counter] = current_node.subset_data(X)
            leaf_vector[leaf_counter] = current_node.expected_value
            leaf_counter += 1
        else:
            visit_nodes.append(current_node.left)
            visit_nodes.append(current_node.right)
        iter_count += 1 
    
    return leaf_subsets, leaf_vector

def tree_predict(X, root_node):
    leaf_subsets, leaf_vector = leaf_extract(X, root_node)
    y_hat = leaf_vector[np.argmax(leaf_subsets, axis = 1)]
    return y_hat

if __name__ == "__main__":
    # Generate covariates
    n = 5000
    p = 10
    rng = np.random.default_rng(1234)
    X = rng.uniform(low = -1, high = 1, size = (n,p))

    # Expected outcome as a monotone step function
    b_0 = 2; b_1 = 1
    E_y = (b_0 + b_1*(X[:,0] > -0.75) + b_1*(X[:,0] > -0.5) + b_1*(X[:,0] > -0.25) + 
        b_1*(X[:,0] > 0) + b_1*(X[:,0] > 0.25) + b_1*(X[:,0] > 0.5) + b_1*(X[:,0] > 0.75))

    # Add gaussian noise
    epsilon = rng.normal(loc = 0, scale = 0.05*np.std(E_y), size = n)

    # Construct outcome
    y = E_y + epsilon

    # Plot y against X, coloring by E_y
    # plt.scatter(X[:,0], y, c = E_y)
    # plt.show()

    # Train the tree
    # Initialize a root node
    root = Node()

    # Compute the expected value at the root node
    root.expected_value = np.mean(y)

    # Add the root node to a BFS queue
    split_queue = deque()
    split_queue.append(root)

    # Search through nodes in the queue and split when the splitting criteria is met
    MIN_INCREASE = 0.0001*np.sum(np.power(y-np.mean(y), 2))
    MAX_ITER = 10000
    iter_count = 0
    while len(split_queue) > 0 and iter_count < MAX_ITER:
        # Select the first node in the queue (FIFO)
        current_node = split_queue.popleft()
        subset_indices = current_node.subset_data(X)
        X_subset = X[subset_indices,:]
        y_subset = y[subset_indices]
        feature_splits = np.array([best_split(X_subset, y_subset, i) for i in range(p)])
        feature_to_split = np.argmax(feature_splits[:,1])

        # Determine whether a proposed split meets the splitting criterion
        # (Reducing sum of squares by more than MIN_INCREASE)
        if feature_splits[feature_to_split,1] > MIN_INCREASE:
            # Create left and right nodes
            left_node = Node()
            right_node = Node()
            
            # Update the current node
            current_node.left = left_node
            current_node.right = right_node
            current_node.split_variable = feature_to_split
            current_node.split_point = feature_splits[feature_to_split,0]
            current_node.node_type = 0

            # Extract the subset condition for the new split
            split_value = feature_splits[feature_to_split,0]
            split_subset_cond_left = X_subset[:,feature_to_split] <= split_value
            split_subset_cond_right = X_subset[:,feature_to_split] > split_value

            # Update the left node
            left_node.parent = current_node
            left_node.expected_value = np.mean(y_subset[split_subset_cond_left])
            left_node.node_origin_type = -1
            left_node.node_type = 1

            # Update the right node
            right_node.parent = current_node
            right_node.expected_value = np.mean(y_subset[split_subset_cond_right])
            right_node.node_origin_type = 1
            right_node.node_type = 1

            # Enqueue the new left and right nodes
            split_queue.append(left_node)
            split_queue.append(right_node)
        
        iter_count += 1

    # Get the predicted values for every sample in X
    y_hat = tree_predict(X, root)

    # Plot y against X, coloring by y_hat
    plt.scatter(X[:,0], y, c = y_hat)
    plt.show()

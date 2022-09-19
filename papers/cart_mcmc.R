###############################################################################
# Simple base R implementation of Bayesian CART Model Search 
# (Chipman, George, McCulloch 1998)
#
# The details below come from the original 1998 paper as well as a 
# Wiley StatsRef overview paper "Bayesian Additive Regression Trees (BART)" 
# (Sparapani and McCulloch 2021)
# 
# The Bayesian approach to constructing CART trees is as follows:
#   1. Initialize tree <- a root node with no splits
#   2. Run m steps of the following algorithm
#      2a. Determine which k of the following four modifications is possible from tree
#         2b.i.   GROW: Randomly choose a leaf node and split it
#                       --> This is always possible
#         2b.ii.  PRUNE: Randomly choose a parent node of two leaf nodes and remove the leaves
#                       --> This is only possible if tree depth > 1
#         2b.iii. SWAP: Randomly choose a pair of interior parent / child nodes and swap their rules
#                       --> This is only possible if tree depth > 2
#         2b.iv.  CHANGE: Randomly choose an interior node and reassign its decision rule
#                       --> This is only possible if tree depth > 1
#      2b. Randomly sample one of the k moves enumerated in 2a
#      2c. Randomly sample an action consistent with the selected rule
#         2c.i.   GROW: Consists of two steps
#              2c.i.1.   Randomly select a feature i
#              2c.i.2.   Randomly select a split point j (assume continuous features)
#         2c.ii.  PRUNE: Only one action - removing the two leaf nodes
#         2c.iii. SWAP: Only one action - swapping the two rules
#         2c.iv.  CHANGE: As with grow, this consists of two steps
#              2c.iv.1.   Randomly select a feature i
#              2c.iv.2.   Randomly select a split point j (assume continuous features)
#      2d. Evaluate the proposed tree against the current tree via Metropolis-Hastings
#         2d.i.   GROW: The only difference between tree and proposed_tree is 
#                       that one of tree's bottom nodes is split into two leaves
#                       so the prior across all tree nodes will be the same except
#                       in the original tree, the split node has probability 
#                       (1 - p_grow(node)) since it wasn't yet split, and
#                       in the proposed tree, the split node has probability
#                       p_grow(node) and its two leaves have probability 
#                       (1 - p_grow(node_left))(1 - p_grow(node_right)). 
# 
#                       Furthermore, there is a probability associated with 
#                       the splitting rule chosen for node, which we call p(rule). 
#                       The marginal likelihood (Y | T, X) differs only in that 
#                       it is aggregated across node in the original tree and 
#                       it is evaluated separately on node_left and node_right 
#                       in the proposed tree.
# 
#                       Finally, the proposal distributions have a ratio of 
#                       p(death)p(internal node)/p(birth)p(bottom node)p(rule)
# 
#                       After canceling identical p(rule) terms, the MH ratio
#                       is p_grow(node)(1 - p_grow(node_left))(1 - p_grow(node_right))p(death)p(internal node)p(y_left | x)p(y_right | x) 
#                       divided by (1 - p_grow(node))p(birth)p(bottom node)p(y_left, y_right | x)
#         2d.ii.  PRUNE: The only difference between tree and proposed_tree is 
#                       that two of tree's bottom nodes are pruned into their parent node
#                       so the MH ratio is the reciprocal of the MH ratio for the GROW step, 
#                       (1 - p_grow(node))p(birth)p(bottom node)p(y_left, y_right | x) divided by
#                       p_grow(node)(1 - p_grow(node_left))(1 - p_grow(node_right))p(death)p(internal node)p(y_left | x)p(y_right | x) 
###############################################################################

###############################################################################
# 1. Data structure functions
#
# Inspired by the array-based implementation of the binary tree data structure 
# on wikipedia: https://en.wikipedia.org/wiki/Binary_tree#Arrays
###############################################################################

# Initialize tree structure as array (to be resized as needed during training)
# Add a root node with sufficient statistics computed for the entire dataset
# (colums 1, 10, 11, and 12) and also set depth = 0 for the root node, 
# node origin type = 0 and node type = 1
tree_init <- function(y, mu_bar, a, nu, lambda, init_size = 10){
  # Col 1: y_bar_j: average of y values that "made it" to a given node j
  # Col 2: left node index
  # Col 3: right node index
  # Col 4: split variable
  # Col 5: split point
  # Col 6: parent node index
  # Col 7: node origin type (0 = root, 1 = right, -1 = left)
  # Col 8: node type (0 = interior, 1 = leaf)
  # Col 9: node depth
  # Col 10: n_j: number of observations that "made it to a given node j"
  # Col 11: s_j: sum of (y_i - y_bar_j) for all y_i observations at node j
  # Col 12: t_j: scaled squared difference between prior mean and node mean
  
  # Create the array
  array_cols = 12
  tree_array = matrix(NA, nrow = init_size, ncol = array_cols)
  
  # Compute sufficient statistics
  y_bar_j = mean(y)
  n_j = length(y)
  s_j = sum((y - y_bar_j)^2)
  t_j = ((n_j*a)/(n_j + a))*(y_bar_j - mu_bar)^2
  
  # Add node information, including sufficient statistics and depth
  tree_array[1,1] = y_bar_j
  tree_array[1,7] = 0
  tree_array[1,8] = 1
  tree_array[1,9] = 0
  tree_array[1,10] = n_j
  tree_array[1,11] = s_j
  tree_array[1,12] = t_j
  
  # Package and return the result
  return(tree_array)
}

# Crawl through a tree from a specific node, 
# return true for all x that reach node_index
get_subset = function(node_index, tree_array, x){
  subset_cond = rep(T, nrow(x))
  current_node_index = node_index
  parent_index = tree_array[current_node_index, 6]
  if (!is.na(parent_index)){
    while(!is.na(parent_index)){
      parent_cutoff = tree_array[parent_index, 5]
      split_variable = tree_array[parent_index, 4]
      child_node_type = tree_array[current_node_index, 7]
      if (child_node_type == 1) {
        child_node_cond <- (x[,split_variable] >= parent_cutoff)
      } else{ 
        if (child_node_type == -1) {
          child_node_cond <- (x[,split_variable] < parent_cutoff)
        } else {
          child_node_cond = subset_cond
        }}
      subset_cond = subset_cond * child_node_cond
      current_node_index = parent_index
      parent_index = tree_array[current_node_index, 6]
    }
  }
  return(subset_cond == 1)
}

# Tree leaf conditions as indicator basis functions
tree_basis = function(tree_array, x){
  # Extract all of the leaves
  leaf_indices = (1:nrow(tree_array))[(tree_array[,8] == 1) & (!is.na(tree_array[,7]))]
  basis_matrix = matrix(NA, nrow = nrow(x), ncol = length(leaf_indices))
  for (i in 1:length(leaf_indices)){
    basis_matrix[,i] = get_subset(leaf_indices[i], tree_array, x)*1
  }
  return(basis_matrix)
}

# Print path to a specific node
node_path = function(node_index, tree_array, x){
  subset_statements = paste0("Node ", format(node_index, digits = 0))
  current_node_index = node_index
  parent_index = tree_array[current_node_index, 6]
  if (!is.na(parent_index)){
    while(!is.na(parent_index)){
      parent_cutoff = tree_array[parent_index, 5]
      split_variable = tree_array[parent_index, 4]
      child_node_type = tree_array[current_node_index, 7]
      if (child_node_type == 1) {
        child_node_statement <- paste0("x[,", split_variable, "] >= ", format(parent_cutoff, digits = 2))
      } else{ 
        if (child_node_type == -1) {
          child_node_statement <- paste0("x[,", split_variable, "] < ", format(parent_cutoff, digits = 2))
        } else {
          child_node_statement = ""
        }}
      subset_statements <- c(subset_statements, child_node_statement)
      current_node_index = parent_index
      parent_index = tree_array[current_node_index, 6]
    }
  }
  subset_statement = paste0(subset_statements, collapse = "  ")
  return(subset_statement)
}

# Print path to each tree leaf
tree_paths = function(tree_array, x){
  # Extract all of the leaves
  leaf_indices = (1:nrow(tree_array))[(tree_array[,8] == 1) & (!is.na(tree_array[,7]))]
  for (i in 1:length(leaf_indices)){
    print(node_path(leaf_indices[i], tree_array, x))
  }
}

# Return predicted values from the tree
predict_tree_leaf = function(tree_array, x){
  # Convert the leaves of the tree into "basis functions"
  X_tree = tree_basis(tree_array, x)
  return(apply(X_tree, 1, function(x) which(x == 1)))
}

# Return a list of all "leaves" in a tree
leaves = function(tree_array){
  # Extract list of leaves
  leaf_indices = (1:nrow(tree_array))[(tree_array[,8] == 1) & (!is.na(tree_array[,7]))]
  if (length(leaf_indices) == 0){
    return(NULL)
  } else{
    return(leaf_indices)
  }
}

# Return a list of all "nog" nodes - - nodes which are parents of leaf nodes
leaf_parents = function(tree_array){
  # Extract list of leaves
  leaf_indices = (1:nrow(tree_array))[(tree_array[,8] == 1) & (!is.na(tree_array[,7]))]
  # Determine the parents of each leaf
  if (all(is.na(tree_array[leaf_indices, 6]))){
    return(NULL)
  } else{
    # Extract list of parent nodes
    parent_nodes = sort(unique(tree_array[leaf_indices, 6]))
    # Filter to nodes for which both left and right are leaves
    cond_1 = tree_array[tree_array[parent_nodes,2],8] == 1
    cond_2 = tree_array[tree_array[parent_nodes,3],8] == 1
    if (sum(cond_1 & cond_2) == 0){
      return(NULL)
    } else{
      return(sort(unique(parent_nodes[cond_1 & cond_2])))
    }
  }
}

# Return a list of all internal nodes - - nodes which have split rules
internal_nodes = function(tree_array){
  # Extract list of internal nodes
  split_node_indices = (1:nrow(tree_array))[!is.na(tree_array[,4])]
  # Return null if the list is empty
  if (length(split_node_indices) > 0){
    return(split_node_indices)
  } else{
    return(NULL)
  }
}

# Return a list of all pairs of internal nodes
internal_node_pairs = function(tree_array){
  # Extract list of internal nodes
  split_node_indices = (1:nrow(tree_array))[!is.na(tree_array[,4])]
  # Restrict to nodes that also have internal nodes as children
  internal_node_left_indices = tree_array[split_node_indices,2]
  internal_node_right_indices = tree_array[split_node_indices,3]
  left_internal = tree_array[internal_node_left_indices,8] == 0
  right_internal = tree_array[internal_node_right_indices,8] == 0
  # Valid internal, left pairs
  if (sum(left_internal) > 1){
    left_pairs = cbind(
      split_node_indices[left_internal], 
      internal_node_left_indices[left_internal]
    )
  } else if (sum(left_internal) == 1) {
    left_pairs = c(
      split_node_indices[left_internal], 
      internal_node_left_indices[left_internal]
    )
  } else{
    left_pairs = NULL
  }
  
  # Valid internal, right pairs
  if (sum(right_internal) > 1){
    right_pairs = cbind(
      split_node_indices[right_internal], 
      internal_node_right_indices[right_internal]
    )
  } else if (sum(right_internal) == 1) {
    right_pairs = c(
      split_node_indices[right_internal], 
      internal_node_right_indices[right_internal]
    )
  } else{
    right_pairs = NULL
  }
  
  if (is.null(left_pairs) & is.null(right_pairs)){
    return(NULL)
  } else{
    valid_pairs = unname(rbind(left_pairs, right_pairs))
    return(valid_pairs)
  }
}

# Prune a tree by removing node leaves 
# Assumes (and checks) that node has two children and both are leaves
prune_node = function(node_index, tree_array){
  # Check that node has children
  stopifnot(tree_array[node_index, 8] == 0)
  
  # Check that node's children are both leaves
  node_left = tree_array[node_index, 2]
  node_right = tree_array[node_index, 3]
  stopifnot(tree_array[node_left, 8] == 1)
  stopifnot(tree_array[node_right, 8] == 1)
  
  # Convert node_index from interior to leaf node
  tree_array[node_index,2] = NA
  tree_array[node_index,3] = NA
  tree_array[node_index,4] = NA
  tree_array[node_index,5] = NA
  tree_array[node_index,8] = 1
  
  # Remove left and right from the tree array
  tree_array[node_left,] = NA
  tree_array[node_right,] = NA
  return(tree_array)
}

# Grow the tree by splitting a leaf node
grow_node <- function(node_index, tree_array, X, y, 
                      split_feature, split_value, a, mu_bar){
  # See how much storage is left in the array
  num_blanks = sum(is.na(tree_array[,1]))
  
  # Resize array if needed
  if (num_blanks < 2){
    tree_array = rbind(tree_array, matrix(NA, nrow = nrow(tree_array), ncol = ncol(tree_array)))
  }
  
  # Subset the data to the current node
  subset_inds = get_subset(node_index, tree_array, X)
  X_subset = X[subset_inds, ]
  y_subset = y[subset_inds]
  
  # Indices at which to store the children nodes
  left_node_index = which(is.na(tree_array[,1]))[1]
  right_node_index = which(is.na(tree_array[,1]))[2]
  
  # Update the array for the splitting node to note which variables were split
  tree_array[node_index,2] = left_node_index
  tree_array[node_index,3] = right_node_index
  tree_array[node_index,4] = split_feature
  tree_array[node_index,5] = split_value
  tree_array[node_index,8] = 0
  
  # Compute left node sufficient statistics
  stopifnot(split_feature %in% 1:ncol(X))
  left_cond = X[,split_feature] <= split_value
  y_left = y[subset_inds & left_cond]
  y_bar_left = mean(y_left)
  n_left = length(y_left)
  s_left = sum((y_left - y_bar_left)^2)
  t_left = ((n_left*a)/(n_left + a))*(y_bar_left - mu_bar)^2
  
  # Update the array for the left node
  tree_array[left_node_index,1] = y_bar_left
  tree_array[left_node_index,6] = node_index
  tree_array[left_node_index,7] = -1
  tree_array[left_node_index,8] = 1
  tree_array[left_node_index,9] = tree_array[node_index,9] + 1
  tree_array[left_node_index,10] = n_left
  tree_array[left_node_index,11] = s_left
  tree_array[left_node_index,12] = t_left
  
  # Compute right node sufficient statistics
  right_cond = X[,split_feature] > split_value
  y_right = y[subset_inds & right_cond]
  y_bar_right = mean(y_right)
  n_right = length(y_right)
  s_right = sum((y_right - y_bar_right)^2)
  t_right = ((n_right*a)/(n_right + a))*(y_bar_right - mu_bar)^2
  
  # Update the array for the right node
  tree_array[right_node_index,1] = y_bar_right
  tree_array[right_node_index,6] = node_index
  tree_array[right_node_index,7] = 1
  tree_array[right_node_index,8] = 1
  tree_array[right_node_index,9] = tree_array[node_index,9] + 1
  tree_array[right_node_index,10] = n_right
  tree_array[right_node_index,11] = s_right
  tree_array[right_node_index,12] = t_right
  
  return(tree_array)
}

# Swap parent_node and child_node split rules
swap_nodes <- function(parent_node, child_node, tree_array, X, y, a, mu_bar){
  # Exchange split rules between parent_node and child_node
  parent_split_feature = tree_array[parent_node,4]
  parent_split_value = tree_array[parent_node,5]
  child_split_feature = tree_array[child_node,4]
  child_split_value = tree_array[child_node,5]
  new_tree_array = tree_array
  new_tree_array[parent_node,4] = child_split_feature
  new_tree_array[parent_node,5] = child_split_value
  new_tree_array[child_node,4] = parent_split_feature
  new_tree_array[child_node,5] = parent_split_value
  
  # Recompute sufficient statistics for all nodes downstream of parent_node
  new_tree_array = recompute_sufficient_statistics(parent_node, new_tree_array, X, y, a, mu_bar, 10000)
  return(new_tree_array)
}

# Change the split rule used for node_index
change_node <- function(node_index, tree_array, new_split_feature, 
                        new_split_value, X, y, a, mu_bar){
  # Reassign split rule for node_index
  tree_array[node_index,4] = new_split_feature
  tree_array[node_index,5] = new_split_value
  
  # Recompute sufficient statistics for all nodes downstream of node_index
  new_tree_array = recompute_sufficient_statistics(node_index, tree_array, X, y, a, mu_bar, 10000)
  return(new_tree_array)
}

# Check if a tree that was modified with a change / swap has observations
# in each node - return True if so, and False otherwise
valid_tree <- function(tree_array){
  node_list = (1:nrow(tree_array))[!is.na(tree_array[,9])]
  if (sum(tree_array[node_list, 10] <= 0) == 0){
    return(T)
  } else{
    return(F)
  }
}

# Recompute and store sufficient statistics for all nodes downstream of (and including) node_index
recompute_sufficient_statistics <- function(node_index, tree_array, X, y, a, mu_bar, MAX_ITER){
  new_tree_array = tree_array
  # Enqueue node_index
  split_queue = list(node_index)
  # Tracking iterations of the algorithm
  iter = 0
  while (length(split_queue) > 0 & iter < MAX_ITER){
    # Pop the current node from the queue
    current_node_index = split_queue[[1]]
    split_queue = split_queue[-1]
    
    # Extract subset up to node_index
    node_subset_inds = get_subset(current_node_index, tree_array, X)
    
    # Compute the sufficient statistics for current_node_index
    y_subset = y[node_subset_inds]
    y_bar_subset = mean(y_subset)
    n_subset = length(y_subset)
    s_subset = sum((y_subset - y_bar_subset)^2)
    t_subset = ((n_subset*a)/(n_subset + a))*((y_bar_subset - mu_bar)^2)
    new_tree_array[current_node_index,1] = y_bar_subset
    new_tree_array[current_node_index,10] = n_subset
    new_tree_array[current_node_index,11] = s_subset
    new_tree_array[current_node_index,12] = t_subset
    
    # Extract left and right node indices
    left_index = tree_array[current_node_index,2]
    right_index = tree_array[current_node_index,3]
    
    # Update the tree array and append to the queue
    if (!is.na(left_index) & !is.na(right_index)){
      # Add the left and right nodes to the queue
      split_queue[length(split_queue)+1] = left_index
      split_queue[length(split_queue)+1] = right_index
    }
    # Increment iterations of the algorithm
    iter = iter + 1
  }
  return(new_tree_array)
}

##################
# The next three functions are for a minimal implementation of CART
# which is helpful for setting prior hyperparameters for Bayesian CART
##################

# (SST - SSL - SSR) formula from rpart documentation
# https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf
variance_reduction = function(cutoff, x, y){
  # Original sum of squares
  ybar_t = mean(y)
  ss_t = sum((y-ybar_t)^2)
  
  # New sum of squares
  cond = (x < cutoff)
  ybar_l = mean(y[cond])
  ss_l = sum((y[cond]-ybar_l)^2)
  ybar_r = mean(y[!cond])
  ss_r = sum((y[!cond]-ybar_r)^2)
  
  return(ss_t - ss_l - ss_r)
}

# For a given restricted set of x and y, find the optimal x cutoff at which to split
tree_split = function(x, y, epsilon = EPSILON){
  # Quit with null value if X is not a matrix with >1 row
  if (is.null(nrow(x))){
    return(NULL)
  }
  
  # Look for optimal splits along each variable
  p = ncol(x)
  optim_results = sapply(1:p, function(j) optimize(function(i) variance_reduction(i, x[,j], y), range(x[,j]), maximum = T))
  optim_results = matrix(as.numeric(optim_results), ncol = p, byrow = F)
  
  # Choose the variable with the largest reduction in variance
  best_col = (1:p)[which(optim_results[2,]==max(optim_results[2,]))[1]]
  
  if (optim_results[2,best_col] > epsilon) return(c(best_col, optim_results[1,best_col]))
  else return(NULL)
}

# Train a decision tree from root, without pruning
fit_tree = function(tree_array, x, y, EPSILON, MAX_ITER){
  # Enqueue the root node
  split_queue = list(1)
  # Tracking iterations of the algorithm
  iter = 0
  while (length(split_queue) > 0 & iter < MAX_ITER){
    # Pop the current node from the queue
    current_node_index = split_queue[[1]]
    split_queue = split_queue[-1]

    # Get the subsetting criteria at this node
    node_subset = get_subset(current_node_index, tree_array, x)
    x_node = x[node_subset,]
    y_node = y[node_subset]
    
    # Search over all variables for a splitting rule
    split_value = tree_split(x_node, y_node, EPSILON)

    # Update the tree array and append to the queue
    if (!is.null(split_value)){
      # See how much storage is left in the array
      num_blanks = sum(is.na(tree_array[,1]))
      
      # Resize array if needed
      if (num_blanks < 2){
        tree_array = rbind(tree_array, matrix(NA, nrow = nrow(tree_array), ncol = ncol(tree_array)))
      }
      
      # Indices at which to store the children nodes
      left_node_index = which(is.na(tree_array[,1]))[1]
      right_node_index = which(is.na(tree_array[,1]))[2]
      
      # Update the array for the splitting node to note which variables were split
      tree_array[current_node_index,2] = left_node_index
      tree_array[current_node_index,3] = right_node_index
      tree_array[current_node_index,4] = split_value[1]
      tree_array[current_node_index,5] = split_value[2]
      tree_array[current_node_index,8] = 0
      
      # Update the array for the left node
      tree_array[left_node_index,1] = mean(y_node[x_node[,split_value[1]] < split_value[2]])
      tree_array[left_node_index,6] = current_node_index
      tree_array[left_node_index,7] = -1
      tree_array[left_node_index,8] = 1
      
      # Update the array for the right node
      tree_array[right_node_index,1] = mean(y_node[x_node[,split_value[1]] >= split_value[2]])
      tree_array[right_node_index,6] = current_node_index
      tree_array[right_node_index,7] = 1
      tree_array[right_node_index,8] = 1
      
      # Add the left and right nodes to the queue
      split_queue[length(split_queue)+1] = left_node_index
      split_queue[length(split_queue)+1] = right_node_index
    }
    # Increment iterations of the algorithm
    iter = iter + 1
  }
  return(tree_array)
}

###############################################################################
# 2. Posterior sampling functions
#
# There are three stages to the posterior sampling:
#    I.   Proposing a tree and accepting / rejecting by Metropolis-Hastings
#    II.  Sampling each of the terminal node means mu_i by Gibbs
#    III. Sampling the global variance parameter sigma^2 by Gibbs
###############################################################################

sample.move <- function(tree_array){
  # Randomly sample an available move
  available_moves = c("GROW")
  if (!is.null(leaf_parents(tree_array))){
    available_moves = c(available_moves, "PRUNE")
  }
  if (!is.null(internal_nodes(tree_array))){
    available_moves = c(available_moves, "CHANGE")
  }
  if (!is.null(internal_node_pairs(tree_array))){
    available_moves = c(available_moves, "SWAP")
  }
  move_probs = c(0.5, 0.4, 0.1)
  move_names = c("GROW/PRUNE", "CHANGE", "SWAP")
  moves_avail = c(T, "CHANGE" %in% available_moves, "SWAP" %in% available_moves)
  move_draw = sample(x = move_names[moves_avail], size = 1, replace = F, 
                     prob = move_probs[moves_avail]/sum(move_probs[moves_avail]))
  if (move_draw == "GROW/PRUNE"){
    if ("PRUNE" %in% available_moves){
      new_draw = sample(x = c("GROW", "PRUNE"), size = 1, replace = F, prob = c(0.5, 0.5))
      return(new_draw)
    } else{
      return("GROW")
    }
  } else{
    return(move_draw)
  }
}

sample.split.node <- function(tree_array){
  leaf_nodes = leaves(tree_array)
  return(sample(leaf_nodes, 1))
}

sample.new.split.rule <- function(X_subset, p){
  if (is.null(dim(X_subset))){
    return(NULL)
  } else{
    # Randomly sample an available splitting feature
    available_features = (1:p)[(apply(X_subset, 2, function(x) length(unique(x))) > 1)]
    if (length(available_features) >= 1){
      if (length(available_features) > 1){
        split_feature = sample(available_features, 1)
      } else{
        split_feature = available_features[1]
      }
    } else{
      return(NULL)
    }
    
    # Randomly sample an available splitting rule
    x_feature = X_subset[,split_feature]
    available_splits = sort(unique(x_feature))[-length(unique(x_feature))]
    split_value = sample(available_splits, 1)
    
    return(c(split_feature, split_value))
  }
}

sample.prune.node <- function(tree_array){
  # List of nog nodes
  nog_nodes = leaf_parents(tree_array)
  if (length(nog_nodes) > 1) out <- sample(nog_nodes, 1)
  else out <- nog_nodes[1]
  return(out)
}

sample.swap.nodes <- function(tree_array){
  # List of internal node pairs
  swap_pairs = internal_node_pairs(tree_array)
  return(swap_pairs[sample(1:nrow(swap_pairs), 1),])
}

sample.change.node <- function(tree_array){
  # List of internal nodes
  change_nodes = internal_nodes(tree_array)
  return(sample(change_nodes, 1))
}

tree.log.likelihood.ratio <- function(original_tree, proposed_tree, X, y, a, mu_bar, nu, lambda){
  leaf_indices_orig = leaves(original_tree)
  n_orig = original_tree[leaf_indices_orig, 10]
  s_orig = original_tree[leaf_indices_orig, 11]
  t_orig = original_tree[leaf_indices_orig, 12]
  n = sum(n_orig)
  ll_original = (1/2)*sum(log((a/(a+n_orig)))) - ((n+nu)/2)*log((sum(s_orig) + sum(t_orig) + nu*lambda))
  
  leaf_indices_prop = leaves(proposed_tree)
  n_prop = proposed_tree[leaf_indices_prop, 10]
  s_prop = proposed_tree[leaf_indices_prop, 11]
  t_prop = proposed_tree[leaf_indices_prop, 12]
  ll_proposed = (1/2)*sum(log((a/(a+n_prop)))) - ((n+nu)/2)*log((sum(s_prop) + sum(t_prop) + nu*lambda))
  
  return(ll_proposed - ll_original)
}

marginal.likelihood.combined <- function(node_index, tree_array, X, y, a, mu_bar, nu, lambda){
  y_bar_j = tree_array[node_index, 1]
  n_j = tree_array[node_index, 10]
  s_j = tree_array[node_index, 11]
  t_j = tree_array[node_index, 12]
  return(((a/(a+n_j))^(1/2))*((s_j + t_j + nu*lambda)^(-(n_j+lambda)/2)))
}

marginal.likelihood.split <- function(left_node_index, right_node_index, tree_array, X, y, a, mu_bar, nu, lambda){
  n <- length(y)
  y_bar_left = tree_array[left_node_index, 1]
  n_left = tree_array[left_node_index, 10]
  s_left = tree_array[left_node_index, 11]
  t_left = tree_array[left_node_index, 12]
  
  y_bar_right = tree_array[right_node_index, 1]
  n_right = tree_array[right_node_index, 10]
  s_right = tree_array[right_node_index, 11]
  t_right = tree_array[right_node_index, 12]
  
  factor_1 = ((a/(a+n_left))^(1/2))*((a/(a+n_right))^(1/2))
  factor_2 = ((s_right + t_right + s_left + t_left + nu*lambda)^(-(n_left+n_right+lambda)/2))
  
  return(factor_1*factor_2)
}

p_split <- function(node_index, tree_array, alpha, beta){
  node_depth = tree_array[node_index, 9]
  return(alpha/((1+node_depth)^(beta)))
}

p_prune <- function(tree_array){
  # Determine the probability of pruning a tree
  available_moves = c("GROW")
  if (!is.null(leaf_parents(tree_array))){
    available_moves = c(available_moves, "PRUNE")
  }
  if (!is.null(internal_nodes(tree_array))){
    available_moves = c(available_moves, "CHANGE")
  }
  if (!is.null(internal_node_pairs(tree_array))){
    available_moves = c(available_moves, "SWAP")
  }
  move_probs = c(0.5, 0.4, 0.1)
  moves_avail = c(T, "CHANGE" %in% available_moves, "SWAP" %in% available_moves)
  move_probs_scaled = move_probs[moves_avail]/sum(move_probs[moves_avail])
  if ("PRUNE" %in% available_moves){
    return(move_probs_scaled[1]/2)
  } else{
    return(NULL)
  }
}

p_grow <- function(tree_array){
  # Determine the probability of pruning a tree
  available_moves = c("GROW")
  if (!is.null(leaf_parents(tree_array))){
    available_moves = c(available_moves, "PRUNE")
  }
  if (!is.null(internal_nodes(tree_array))){
    available_moves = c(available_moves, "CHANGE")
  }
  if (!is.null(internal_node_pairs(tree_array))){
    available_moves = c(available_moves, "SWAP")
  }
  move_probs = c(0.5, 0.4, 0.1)
  moves_avail = c(T, "CHANGE" %in% available_moves, "SWAP" %in% available_moves)
  move_probs_scaled = move_probs[moves_avail]/sum(move_probs[moves_avail])
  if ("PRUNE" %in% available_moves){
    return(move_probs_scaled[1]/2)
  } else{
    return(move_probs_scaled[1])
  }
}

p_rule <- function(X_subset, p){
  # Number of unique splitting values available on X_subset
  if (!is.null(dim(X_subset))){
    available_features = (1:p)[(apply(X_subset, 2, function(x) length(unique(x))) > 1)]
    num_rules = sum(apply(X_subset, 2, function(x) length(unique(x))))
    if (num_rules > 0) return(1/num_rules)
    else return(0)
  } else{
    return(0)
  }
}

log_p_tree_node <- function(node_index, tree_array, alpha, beta, X, y, MAX_ITER){
  p <- ncol(X)
  log_sum <- 0
  # Enqueue node_index
  split_queue = list(node_index)
  # Tracking iterations of the algorithm
  iter = 0
  while (length(split_queue) > 0 & iter < MAX_ITER){
    # Pop the current node from the queue
    current_node_index = split_queue[[1]]
    split_queue = split_queue[-1]
    
    # Extract subset up to node_index
    node_subset_inds = get_subset(current_node_index, tree_array, X)
    X_subset = X[node_subset_inds,]
    
    # Compute p_node
    is_internal = tree_array[current_node_index,8] == 0
    p_node <- is_internal*(p_rule(X_subset, p)*p_split(current_node_index, tree_array, alpha, beta)) + (!is_internal)*(1-p_split(current_node_index, tree_array, alpha, beta))

    # Add to the result
    log_sum <- log_sum + log(p_node)

    # Extract left and right node indices
    left_index = tree_array[current_node_index,2]
    right_index = tree_array[current_node_index,3]
    
    # Update the tree array and append to the queue
    if (!is.na(left_index) & !is.na(right_index)){
      # Add the left and right nodes to the queue
      split_queue[length(split_queue)+1] = left_index
      split_queue[length(split_queue)+1] = right_index
    }
    # Increment iterations of the algorithm
    iter = iter + 1
  }
  return(log_sum)
}

mh.tree <- function(tree_array, X, y, a, mu_bar, nu, lambda, alpha, beta){
  # Randomly sample a new mode
  new_move = sample.move(tree_array)
  print(paste0("NEW MOVE = ", new_move))
  if (new_move == "GROW"){
    node_to_split <- sample.split.node(tree_array)
    if (verbose){
      print(paste0("GROWING NODE: ", node_to_split))
    }
    subset_inds <- get_subset(node_to_split, tree_array, X)
    X_subset = X[subset_inds,]; p <- ncol(X)
    split_rule <- sample.new.split.rule(X_subset, p)
    if (is.null(split_rule)){
      # Automatically reject leaf nodes that can't be split
      log_mh_ratio <- -Inf
    } else{
      split_feature <- split_rule[1]
      split_value <- split_rule[2]
      new_tree_array <- grow_node(node_to_split, tree_array, X, y, split_feature, 
                                  split_value, a, mu_bar)
      new_left_node <- new_tree_array[node_to_split,2]
      new_right_node <- new_tree_array[node_to_split,3]
      PG <- p_split(node_to_split, tree_array, alpha, beta)
      PG.L <- p_split(new_left_node, new_tree_array, alpha, beta)
      PG.R <- p_split(new_right_node, new_tree_array, alpha, beta)
      PD <- p_prune(new_tree_array)
      PB <- p_grow(tree_array)
      Pbot <- 1/length(leaves(tree_array))
      Pnog <- 1/length(leaf_parents(new_tree_array))
      log.lr <- tree.log.likelihood.ratio(tree_array, new_tree_array, X, y, a, mu_bar, nu, lambda)
      log_mh_ratio <- log.lr + log((PD*Pnog)/(PB*Pbot)) + log((PG*(1-PG.L)*(1-PG.R))/(1-PG))
    }
  } else if (new_move == "PRUNE"){
    node_to_prune <- sample.prune.node(tree_array)
    if (verbose){
      print(paste0("PRUNING NODE: ", node_to_prune))
    }
    old_left_node <- tree_array[node_to_prune,2]
    old_right_node <- tree_array[node_to_prune,3]
    new_tree_array <- prune_node(node_to_prune, tree_array)
    PG <- p_split(node_to_prune, new_tree_array, alpha, beta)
    PG.L <- p_split(old_left_node, tree_array, alpha, beta)
    PG.R <- p_split(old_right_node, tree_array, alpha, beta)
    PD <- p_prune(tree_array)
    PB <- p_grow(new_tree_array)
    Pbot <- 1/length(leaves(new_tree_array))
    Pnog <- 1/length(leaf_parents(tree_array))
    log.lr <- tree.log.likelihood.ratio(tree_array, new_tree_array, X, y, a, mu_bar, nu, lambda)
    log_mh_ratio <- log.lr + log((PB*Pbot)/(PD*Pnog)) + log((1-PG)/(PG*(1-PG.L)*(1-PG.R)))
  } else if (new_move == "SWAP"){
    nodes_to_swap <- sample.swap.nodes(tree_array)
    node_to_swap_1 <- nodes_to_swap[1]
    node_to_swap_2 <- nodes_to_swap[2]
    if (verbose){
      print(paste0("SWAPPING NODES: ", node_to_swap_1, " and ", node_to_swap_2))
    }
    new_tree_array <- swap_nodes(node_to_swap_1, node_to_swap_2, tree_array, X, y, a, mu_bar)
    if (!valid_tree(new_tree_array)){
      log_mh_ratio <- -Inf
    } else{
      log.lr <- tree.log.likelihood.ratio(tree_array, new_tree_array, X, y, a, mu_bar, nu, lambda)
      log_prior_orig_tree <- log_p_tree_node(node_to_swap_1, tree_array, alpha, beta, X, y, 10000)
      log_prior_prop_tree <- log_p_tree_node(node_to_swap_1, new_tree_array, alpha, beta, X, y, 10000)
      log_mh_ratio <- log.lr + log_prior_prop_tree - log_prior_orig_tree
    }
  } else if (new_move == "CHANGE"){
    node_to_change <- sample.change.node(tree_array)
    if (verbose){
      print(paste0("CHANGING NODE: ", node_to_change))
    }
    subset_inds <- get_subset(node_to_change, tree_array, X)
    X_subset = X[subset_inds,]; p <- ncol(X)
    split_rule <- sample.new.split.rule(X_subset, p)
    if (is.null(split_rule)){
      # Automatically reject leaf nodes that can't be split
      log_mh_ratio <- -Inf
    } else{
      split_feature <- split_rule[1]
      split_value <- split_rule[2]
      new_tree_array <- change_node(node_to_change, tree_array, split_feature, 
                                    split_value, X, y, a, mu_bar)
      if (!valid_tree(new_tree_array)){
        log_mh_ratio <- -Inf
      } else{
        log.lr <- tree.log.likelihood.ratio(tree_array, new_tree_array, X, y, a, mu_bar, nu, lambda)
        log_prior_orig_tree <- log_p_tree_node(node_to_change, tree_array, alpha, beta, X, y, 10000)
        log_prior_prop_tree <- log_p_tree_node(node_to_change, new_tree_array, alpha, beta, X, y, 10000)
        log_mh_ratio <- log.lr + log_prior_prop_tree - log_prior_orig_tree
      }
    }
  } else {
    new_tree_array <- NULL
  }
  
  # Now perform the MH update
  acceptance_prob = min(1, exp(log_mh_ratio))
  if(runif(1) <= acceptance_prob){
    next_tree = new_tree_array
    decision = "ACCEPT"
  } else{
    next_tree = tree_array
    decision = "REJECT"
  }
  
  return(list(new_tree = next_tree, acceptance_prob = acceptance_prob, 
              decision = decision, move = new_move))
}

mu.sample = function(tree_array, sigma2, a, mu_bar){
  # Extract leaf sufficient statistics
  leaf_inds = leaves(tree_array)
  y_bar_leaves = tree_array[leaf_inds,1]
  n_leaves = tree_array[leaf_inds,10]
  
  # Draw mu_i for each of the leaves
  mu_vec = (y_bar_leaves*n_leaves + a*mu_bar)/(n_leaves + a)
  sigma2_vec = sigma2/(n_leaves + a)
  return(rnorm(length(mu_vec), mean = mu_vec, sd = sqrt(sigma2_vec)))
}

global.sigma.sample = function(tree_array, nu, lambda){
  # Extract leaf sufficient statistics
  leaf_inds = leaves(tree_array)
  s_leaves = tree_array[leaf_inds,11]
  t_leaves = tree_array[leaf_inds,12]
  n_leaves = tree_array[leaf_inds,10]
  n = sum(n_leaves)
  
  # Compute the posterior parameters
  nu_1 = (nu + n)/2
  nu_lambda_1 = (nu*lambda + sum(s_leaves) + sum(t_leaves))/2
  
  # Sample from the IG posterior
  return(1/rgamma(1, shape = nu_1, rate = nu_lambda_1))
}

###############################################################################
# Example of the algorithm
###############################################################################

# Generate data from a simple 1D function
n = 5000
p = 10
X = matrix(runif(n*p, -1, 1), ncol = p)
# # Example 1: Monotone step function
b_0 = 1; b_1 = 1
E_y = (b_0 + b_1*(X[,1] > -0.75) + b_1*(X[,1] > -0.5) + b_1*(X[,1] > -0.25) +
       b_1*(X[,1] > 0) + b_1*(X[,1] > 0.25) + b_1*(X[,1] > 0.5) + b_1*(X[,1] > 0.75))
# Sample random (additive) outcome noise
eps = rnorm(n, 0, 0.1)
y = E_y + eps
# plot(X[,1], y)

# Train an overfit decision tree
greedy_tree_init = tree_init(y, 0, 1, 1, 1, init_size = 10)
EPSILON = 0.001*sum((y-mean(y))^2)
MAX_ITER = 1000
greedy_tree_fit = fit_tree(greedy_tree_init, X, y, EPSILON, MAX_ITER)

# Convert the leaves of the tree into "basis functions"
X_tree = tree_basis(greedy_tree_fit, X)
# Convert the matrix of bases into a single categorical variable
X_basis_cat = apply(X_tree, 1, function(x) which(x == 1)[1])
# Extract "predictions" from the tree
leaf_indices = (1:nrow(greedy_tree_fit))[(greedy_tree_fit[,8] == 1) & (!is.na(greedy_tree_fit[,7]))]
y_hat = greedy_tree_fit[apply(X_tree, 1, function(x) leaf_indices[which(x == 1)]),1]
# Plot the predicted y values, coloring by tree leaf
plot(X[,1], y_hat, col = X_basis_cat)
# Plot the predicted versus actual y values
plot(y, y_hat)

# Set prior parameters
# First, set the IG parameters - nu and lambda so that the prior includes 
# the unconditional empirical variance of y and the empirical variance 
# conditional on an overfit decision tree
sigma_hat_underfit = var(y)
sigma_hat_overfit = var(y-y_hat)
nu = 3
lambda = 1
beta = (nu*lambda)/2; alpha = nu/2
x.seq = seq(0, sigma_hat_underfit, length.out = 10000)
plot(x.seq, (((beta)^alpha) / gamma(alpha))*((1/x.seq)^(alpha+1))*(exp(-(beta/(x.seq)))), 
     col = "blue", lwd = 3, lty = 3, ylab = "Density", xlab = "Sigma", 
     xlim = c(sigma_hat_overfit - 0.5, sigma_hat_underfit + 0.5))
abline(v = sigma_hat_underfit)
abline(v = sigma_hat_overfit)

# Next, set the Normal parameters - mu_bar and a, so that the prior distribution 
# on mu_j covers the range of y
par(mfrow = c(3,2))
n_sigma_draws = 6
mu_bar = mean(c(min(y), max(y)))
a = 1/4
x.seq = seq(min(y), max(y), length.out = 10000)
for (j in 1:n_sigma_draws){
  sigma2 = 1/rgamma(1, shape = nu/2, rate = (nu*lambda)/2)
  plot(x.seq, dnorm(x.seq, mean = mu_bar, sd = sqrt(sigma2/a)), 
       col = "blue", lwd = 3, lty = 3, ylab = "Density", xlab = "Sigma", 
       main = paste0("sigma2 = ", sigma2), xlim = c(min(y), max(y)))
  abline(v = min(y))
  abline(v = max(y))
}

# Initialize a tree
tree_array = tree_init(y, mu_bar, a, nu, lambda, init_size = 10)
leaf_parents(tree_array)
internal_nodes(tree_array)
internal_node_pairs(tree_array)

# Initialize mu and sigma2
sigma2 = ((nu*lambda)/2)/((nu/2) - 1)
mu_vector = c(mu_bar)

# Predict y given mu and sigma
leaf_numbers = predict_tree_leaf(tree_array, X)
y_hat = mu_vector[leaf_numbers]

# Perform one step of the MH-algorithm
alpha = 0.95
beta = 0.5
mh_list = mh.tree(tree_array, X, y, a, mu_bar, nu, lambda, alpha, beta)

# Extract the details of the MH iteration
tree_array = mh_list$new_tree
mh_alpha = mh_list$acceptance_prob
mh_decision = mh_list$decision
mh_move = mh_list$move

# Sample mu and sigma
mu_vec = mu.sample(tree_array, sigma2, a, mu_bar)
sigma2 = global.sigma.sample(tree_array, nu, lambda)

# Predict y given mu and sigma
leaf_numbers = predict_tree_leaf(tree_array, X)
y_hat = mu_vector[leaf_numbers]
par(mfrow = c(1,1))
plot(y, y_hat)

# Perform several more iterations of the algorithm
verbose = T
burnin = 500
mc = 1000
nthin = 20
y_hat_samples <- matrix(NA, nrow = mc/nthin, ncol = n)
for (i in 1:(burnin+mc)){
  # Perform one step of the MH-algorithm
  mh_list = mh.tree(tree_array, X, y, a, mu_bar, nu, lambda, alpha, beta)
  
  # Extract the details of the MH iteration
  tree_array = mh_list$new_tree
  mh_alpha = mh_list$acceptance_prob
  mh_decision = mh_list$decision
  mh_move = mh_list$move
  if (verbose){
    print(paste0("MOVE = ", mh_move))
    print(paste0("DECISION = ", mh_decision))
  }
  
  # Sample mu and sigma
  mu_vec = mu.sample(tree_array, sigma2, a, mu_bar)
  sigma2 = global.sigma.sample(tree_array, nu, lambda)
  
  # Predict y given mu and sigma
  leaf_numbers = predict_tree_leaf(tree_array, X)
  y_hat = mu_vec[leaf_numbers]
  if (i %% nthin == 0){
    plot(y, y_hat)
  }
  if ((i > burnin) & (((i - burnin) %% nthin == 0))){
    y_hat_samples[(i - burnin) %/% nthin,] <- y_hat
  }
}

plot(X[,1], y, pch = 16, cex = 1)
for (i in 1:nrow(y_hat_samples)){
  points(X[,1], y_hat_samples[i,], col = "gray", pch = 16, cex = 0.25)
}

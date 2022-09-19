###############################################################################
# Very simple R-native implementation of CART for continuous outcomes 
# and continuous predictors
###############################################################################

### 1. Data structure functions

# Inspired by the array-based implementation of the binary tree data structure 
# on wikipedia: https://en.wikipedia.org/wiki/Binary_tree#Arrays

# Crawl through a tree from a specific node, 
# extract all of the conditions that are true at that node
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
# Tree leaf conditions as basis functions
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
predict_tree = function(tree_array, x){
  # Convert the leaves of the tree into "basis functions"
  X_tree = tree_basis(tree_array, x)
  # Return the expected values associated with each data point's leaf node
  leaf_indices = (1:nrow(tree_array))[(tree_array[,8] == 1) & (!is.na(tree_array[,7]))]
  return(tree_array[apply(X_tree, 1, function(x) leaf_indices[which(x == 1)]),1])
}

### 2. Model fitting functions

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
# For a given restricted set of x and y, find the optimal x 
# cutoff at which to split
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

### 3. Test the implementation

# Generate data from a simple 1D function
n = 5000
p = 2
x = matrix(runif(n*p, -1, 1), ncol = p)
# # Example 1: Monotone step function
b_0 = 2; b_1 = 1
E_y = (b_0 + b_1*(x[,1] > -0.75) + b_1*(x[,1] > -0.5) + b_1*(x[,1] > -0.25) +
       b_1*(x[,1] > 0) + b_1*(x[,1] > 0.25) + b_1*(x[,1] > 0.5) + b_1*(x[,1] > 0.75))
# # Example 2: Fluctuating step function
# b_0 = 2; b_1 = 1
# E_y = (b_0 + b_1*(x[,1] > -0.75) + b_1*(x[,1] > -0.5) - b_1*(x[,1] > -0.25) -
#        b_1*(x[,1] > 0) + b_1*(x[,1] > 0.25) + b_1*(x[,1] > 0.5) - b_1*(x[,1] > 0.75))
# # Example 3: sin function
# E_y = sin(x[,1]*pi) + sin(x[,2]*pi)
# Sample random (additive) outcome noise
eps = rnorm(n, 0, 0.1)
y = E_y + eps
# plot(x[,1], y)

# Initialize an array of arbitrary length (will be resized as needed during training)
# Col 1: expected value, given the conditions that apply to a given node
# Col 2: left node index
# Col 3: right node index
# Col 4: split variable
# Col 5: split point
# Col 6: parent node index
# Col 7: node origin type (0 = root, 1 = right, -1 = left)
# Col 8: node type (0 = interior, 1 = leaf)
array_init_size = 10
array_cols = 8
tree_array = matrix(NA, nrow = array_init_size, ncol = array_cols)

# Add the root node to the array
tree_array[1,1] = mean(y)

# Train the decision tree
EPSILON = 0.0001*sum((y-mean(y))^2)
MAX_ITER = 1000
tree_array = fit_tree(tree_array, x, y, EPSILON, MAX_ITER)

# Convert the leaves of the tree into "basis functions"
X_tree = tree_basis(tree_array, x)
# Convert the matrix of bases into a single categorical variable
X_basis_cat = apply(X_tree, 1, function(x) which(x == 1)[1])
# Plot the data, color by tree leaf
plot(x[,1], y, col = X_basis_cat)
# Extract "predictions" from the tree
y_hat = predict_tree(tree_array, x)
# Plot the predicted y values, coloring by tree leaf
plot(x[,1], y_hat, col = X_basis_cat)
# Plot the predicted versus actual y values
plot(y, y_hat)
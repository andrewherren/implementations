###############################################################################
# A demonstration of how the conceptually simple machinery of a single hidden 
# layer neural network with ReLU (rectified linear unit) activation function, 
# defined as ReLU(x) = max(0, x) can be used to construct a linear 
# spline for univariate functions (which will interpolate if we choose 
# as many knots as samples)
###############################################################################

# Generate a simple (x, y) sample
n <- 100
x <- runif(n)
y <- sin(12*pi*x)

# Sort x
num_samples_per_knot <- 1
knots <- sort(x)[seq(from = 1, to = length(x)-1, by = num_samples_per_knot)]

# The weights mapping to each of the hidden nodes are fixed at 1
weight1 <- t(rep(1,length(knots)))

# The bias terms are set in at -1*(knots)
bias1 <- t(-knots)

# Use the weight, bias, and relu activation to compute the hidden layer
hidden_layer <- pmax(sweep(x %*% weight1, 2, bias1, FUN = "+"), 0)

# Compute the weights mapping the hidden layer to the output
U <- cbind(1, hidden_layer)
Uty <- t(U) %*% y
UtU <- t(U) %*% U
coef <- solve(UtU, Uty)
weight2 <- coef[2:length(coef)]
bias2 <- coef[1]

# Compute and plot the resulting predictions of the neural network
output <- hidden_layer %*% weight2 + bias2
plot(output, y)

# We can now use this neural network to predict on unseen X values
x_new <- runif(10000)
hidden_layer <- pmax(sweep(x_new %*% weight1, 2, bias1, FUN = "+"), 0)
output <- hidden_layer %*% weight2 + bias2
plot(x_new, output)

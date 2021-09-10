# Linear regression with horseshoe prior, as in Makalic and Schmidt (2015)
# https://arxiv.org/abs/1508.03884

# [NOTE: Carvalho et al (2010) is the original paper introducing the horseshoe 
#  estimator, and many other authors have written about the horseshoe, but
#  Makalic and Schmidt (2015) present a particularly straightforward set of 
#  conditional distributions, making for an easy / didactic Gibbs implementation]

# Load required packages
pkg.list <- c("MASS")
lapply(pkg.list, require, character.only = T)

# Simulated sparse DGP
set.seed(1234)
n <- 100
p <- 20
beta <- c(2, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
sigma_true <- 0.5
epsilon <- rnorm(n, 0, sigma_true)
X <- matrix(runif(n*p, -1, 1), ncol = p, byrow = T)
y <- as.numeric(X %*% beta) + epsilon

## 0. OLS Results
reg.lm <- lm(y ~ X)
summary(reg.lm)

## 1. Manually-implemented Gibbs sampler (using conditional distributions 
##    from Makalic and Schmidt (2015))

# Number of draws
n_sim <- 1000

# Starting point
lambda_0 <- 0.5
lambda_i_0 <- rep(0.5, p)
sigma_0 <- 1
beta_0 <- rep(0, p)

# Empty vector to store draws
lambda_i_samples <- matrix(NA, nrow = n_sim, ncol = p)
lambda_samples <- matrix(NA, nrow = n_sim, ncol = 1)
beta_samples <- matrix(NA, nrow = n_sim, ncol = p)
sigma_samples <- matrix(NA, nrow = n_sim, ncol = 1)

for (i in 1:n_sim){
  # Retrieve the "starting values" of parameters
  if (i == 1){
    beta_init <- beta_0
    sigma_init <- sigma_0
    lambda_i_init <- lambda_i_0
    lambda_init <- lambda_0
  } else{
    beta_init <- beta_samples[i-1,]
    sigma_init <- sigma_samples[i-1,]
    lambda_i_init <- lambda_i_samples[i-1,]
    lambda_init <- lambda_samples[i-1,]
  }
  
  # Draw auxiliary variable for lambda
  # eta ~ IG(1, 1 + 1/lambda^2)
  nu <- 1/rgamma(1, shape = 1, rate = 1 + 1/lambda_init^2)
  
  # Draw auxiliary variable for lambda_i
  # nu_i ~ IG(1, 1 + 1/lambda^2)
  nu_i <- 1/rgamma(p, shape = 1, rate = 1 + 1/lambda_i_init^2)
  
  # Draw lambda
  # lambda ~ IG((p+1)/2, (1/nu) + (1/(2*(sigma_init^2)))*sum((beta_init^2)/(lambda_i_init^2)))
  lambda_updated <- sqrt(1/rgamma(1, shape = (p+1)/2, rate = (1/nu) + (1/(2*(sigma_init^2)))*sum((beta_init^2)/(lambda_i_init^2))))
  
  # Draw lambda_i
  # lambda ~ IG((p+1)/2, (1/nu) + (1/(2*(sigma_init^2)))*sum((beta_init^2)/(lambda_i_init^2)))
  lambda_i_updated <- sqrt(1/rgamma(p, shape = 1, rate = (1/nu_i) + (1/(2*(sigma_init^2)))*((beta_init^2)/(lambda_updated^2))))
  
  # Draw sigma
  # sigma ~ IG((n+p)/2, (y - Xb)'(y - Xb)/2 + b'Lambda^(-1)b/2)
  Lambda <- (lambda_updated^2)*diag(lambda_i_updated^2)
  resid <- y - as.numeric(X %*% beta_init)
  sigma_updated <- sqrt(1/rgamma(1, shape = (n+p)/2, rate = as.numeric((t(resid) %*% resid)/2) + as.numeric((t(beta_init) %*% solve(Lambda) %*% beta_init))/2))
  
  # Draw beta
  # beta ~ N(A^(-1) X'y, sigma^2 A^(-1))
  A <- (t(X) %*% X + solve(Lambda))
  Ai <- solve(A)
  mu_vec <- as.numeric(Ai %*% t(X) %*% y)
  sigma_mat <- sigma_updated^2 * Ai
  beta_updated <- mvrnorm(1, mu = mu_vec, Sigma = sigma_mat)

  # Store results
  beta_samples[i,] <- beta_updated
  lambda_i_samples[i,] <- lambda_i_updated
  lambda_samples[i,] <- lambda_updated
  sigma_samples[i,] <- sigma_updated
}

# Visualize results
col <- 10
hist(beta_samples[,col], freq = F, breaks = 50)
# Draw data
n = 100
p = 20
beta_true = c(c(1,2,3), rnorm(p-3, 0, 0.01))
kappa = 1.25
sigma = kappa * sqrt(sum(beta_true^2))
X = matrix(rnorm(n*p), ncol = p)
y = as.numeric(X %*% beta_true) + sigma * rnorm(n)

# Precompute OLS estimates
beta_hat_ols = as.numeric(solve(t(X) %*% X) %*% t(X) %*% y)

# Precompute sufficient statistics, assuming each coefficient is in a block of its own
XtX = t(X) %*% X
XtXi = solve(XtX)
cond_mean_adj = matrix(NA, nrow = p, ncol = p-1)
cond_var_adj = rep(NA, p)
for (i in 1:p){
  cond_mean_adj[i,] = XtXi[i,-i] %*% solve(XtXi[-i,-i])
  cond_var_adj[i] = as.numeric(XtXi[i,i] - (XtXi[i,-i] %*% solve(XtXi[-i,-i]) %*% XtXi[-i,i]))
}

# Define sampling functions
horseshoe_prior <- function(beta_k, lambda){
  return((1/(2*sqrt(2*(pi^3))))*log(1+(4/((beta_k/lambda)^2))))
}

joint_horseshoe_prior <- function(beta_k, lambda){
  return((1/(2*lambda*sqrt(2*(pi^3))))*log(1+(4/((beta_k/lambda)^2)))*(2/(pi*(1+(lambda^2)))))
}

slice_evaluation <- function(a, b, beta_tilde_k, v0, v1, lambda){
  theta_new = runif(1, a, b)
  return(list(
    pi_eval = horseshoe_prior(beta_tilde_k + v0*sin(theta_new) + v1*cos(theta_new), lambda),
    theta_new = theta_new
  ))
}

beta_k_update <- function(k, beta, sigma2, lambda, theta, cond_mean_adj, cond_var_adj, beta_hat_ols){
  # Step 1
  beta_tilde_k = beta_hat_ols[k] + as.numeric(cond_mean_adj[k,] %*% (beta[-k] - beta_hat_ols[-k]))
  delta_k = beta[k] - beta_tilde_k
  v <- rnorm(1, 0, sqrt(sigma2 * cond_var_adj[k]))
  v0 = delta_k * sin(theta[k]) + v * cos(theta[k])
  v1 = delta_k * cos(theta[k]) - v * sin(theta[k])

  # Step 2: slice sampling delta
  ell <- runif(1, 0, horseshoe_prior(delta_k + beta_tilde_k, lambda))
  a = 0
  b = 2*pi
  slice_eval = slice_evaluation(a, b, beta_tilde_k, v0, v1, lambda)
  new_prior_eval = slice_eval$pi_eval
  theta_new = slice_eval$theta_new
  while(new_prior_eval < ell){
    if (theta_new < theta[k]) {
      a <- theta_new
    } else {
      b <- theta_new
    }
    slice_eval = slice_evaluation(a, b, beta_tilde_k, v0, v1, lambda)
    new_prior_eval = slice_eval$pi_eval
    theta_new = slice_eval$theta_new
  }
  theta_accept = theta_new

  # Extract delta and beta
  delta_accept = v0*sin(theta_accept) + v1*cos(theta_accept)
  beta_accept = beta_tilde_k + delta_accept

  return(list(beta = beta_accept, theta = theta_accept))
}

sigma_update <- function(y, X, beta, alpha, gamma){
  n <- length(y)
  s <- as.numeric(t(y - X %*% beta) %*% (y - X %*% beta))
  s1 <- (n + alpha)/2
  s2 <- (s + gamma)/2
  return(1/rgamma(1, shape = s1, rate = s2))
}

lambda_update <- function(lambda, beta){
  r <- rnorm(1, 0, (0.2))
  lambda_proposed <- exp(log(lambda) + r)
  log_prior_initial_lambda = sum(sapply(1:p, function(i) log(joint_horseshoe_prior(beta[i], lambda))))
  log_prior_proposed_lambda = sum(sapply(1:p, function(i) log(joint_horseshoe_prior(beta[i], lambda_proposed))))
  eta <- exp(log_prior_proposed_lambda - log_prior_initial_lambda + log(lambda_proposed) - log(lambda))
  u <- runif(1)
  return(ifelse(eta > u, lambda_proposed, lambda))
}

# Initialize sampling containers and hyperparameters
betas = rep(1, p)
sigma2 = 1
lambda = 1
theta = rep(1, p)
alpha = 1
gamma = 1
mc = 10000
beta_mc = matrix(NA, nrow = p, ncol = mc)
sigma_mc = rep(NA, mc)
lambda_mc = rep(NA, mc)

# Run the sampler
for (k in 1:mc){
  for (i in 1:p){
    update_list = beta_k_update(i, betas, sigma2, lambda, theta, cond_mean_adj, cond_var_adj, beta_hat_ols)
    betas[i] = update_list$beta
    theta[i] = update_list$theta
  }
  sigma2 = sigma_update(y, X, betas, alpha, gamma)
  lambda = lambda_update(lambda, betas)

  beta_mc[,k] = betas
  sigma_mc[k] = sigma2
  lambda_mc[k] = lambda
}

burnin = 2000
beta_est = rowMeans(beta_mc[,(burnin+1):mc])
plot(NULL,xlim=range(beta_true),ylim=range(c(beta_true, beta_est)),
     xlab = "beta true", ylab = "estimation", )
points(beta_true,beta_est,pch=20)
abline(0,1,col='red')

hist(beta_mc[1,(burnin+1):mc])
abline(v = beta_true[1], col = "red", lty = 3, lwd = 3)
abline(v = beta_hat_ols[1], col = "blue", lty = 3, lwd = 3)

hist(beta_mc[2,(burnin+1):mc])
abline(v = beta_true[2], col = "red", lty = 3, lwd = 3)
abline(v = beta_hat_ols[2], col = "blue", lty = 3, lwd = 3)

hist(beta_mc[3,(burnin+1):mc])
abline(v = beta_true[3], col = "red", lty = 3, lwd = 3)
abline(v = beta_hat_ols[3], col = "blue", lty = 3, lwd = 3)

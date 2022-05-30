###############################################################################
# Simple R implementation of the semiparametric average treatment effect 
# estimators analyzed in Hirano, Imbens, and Ridder 2003 (Estimation of Average 
# Treatment Effects using the Estimated Propensity Score)
# 
# General idea is to use data-adaptive series estimation of the propensity 
# score, and then estimate the ATE using IPW on the fitted propensity scores, 
# with an asymptotic variance estimator derived based on semiparametric theory
#
# The insight is that estimating the ATE using fitted weights (even when the 
# true propensity score is known) attains a lower asymptotic variance under 
# standard regularity conditions, and that a variance estimator can be constructed 
# to adjust for the first-step nonparametric estimation of the propensity score
###############################################################################

n_sim = 10000
sim_results = matrix(NA, nrow = n_sim, ncol = 12)

for (i in 1:n_sim){
  # Sample size
  n = 500
  
  # Covariates
  p = 2
  X = matrix(runif(n*p), ncol = p)
  
  # Treatment
  pi_x = 0.2 + (X[,1]*(1-X[,2]) + X[,2]*(1-X[,1]))*0.6
  Z = rbinom(n, 1, pi_x)
  
  # Outcome 
  tau = 2
  epsilon = rnorm(n, 0, 1)
  E_y = sin(pi*X[,1]) + cos(pi*X[,2]) + X[,1]*X[,2] + tau*Z
  y = E_y + epsilon
  
  # Polynomial basis (the paper assumes that ncol(P_K) is less than n^(1/9), 
  # but we stretch that assumption in order to fit pi_x nicely)
  K = 4
  P_K = cbind(1, poly(X, degree = K))
  
  # Estimate pi_x using the nonparametric series, truncate if necessary 
  # to avoid numeric issues when weighting by pi_hat and 1-pi_hat
  pi_hat = predict(glm(Z ~ 0 + P_K, binomial(link = "logit")), type = "response")
  eps = 0.05
  pi_hat = ifelse(pi_hat < eps, eps, ifelse(pi_hat > 1 - eps, 1 - eps, pi_hat))
  
  # Compute the ATEs
  tau_hat_fitted = mean((y*Z)/(pi_hat)) - mean((y*(1-Z))/(1-pi_hat))
  tau_hat_true = mean((y*Z)/(pi_x)) - mean((y*(1-Z))/(1-pi_x))
  
  # Estimate the asymptotic variances
  phi_hat = ((y*Z)/(pi_hat) - (y*(1-Z))/(1-pi_hat)) - tau_hat_fitted
  alpha_projection = predict(lm((((y*Z)/(pi_hat^2)) + ((y*(1-Z))/((1-pi_hat)^2))) ~ 0 + P_K))
  alpha_hat = - alpha_projection * (Z - pi_hat)
  var_hat_fitted = mean((phi_hat + alpha_hat)^2)
  var_hat_true = mean((((y*Z)/(pi_x) - (y*(1-Z))/(1-pi_x)) - tau_hat_true)^2)
  
  # Compute asymptotic confidence interval for the fitted weights estimator
  ci_lb_fitted = tau_hat_fitted - qnorm(0.975)*sqrt(var_hat_fitted)/sqrt(n)
  ci_ub_fitted = tau_hat_fitted + qnorm(0.975)*sqrt(var_hat_fitted)/sqrt(n)
  coverage_fitted = ((ci_lb_fitted <= tau) & (ci_ub_fitted >= tau))*1
  
  # Compute asymptotic confidence interval for the true weights estimator
  ci_lb_true = tau_hat_true - qnorm(0.975)*sqrt(var_hat_true)/sqrt(n)
  ci_ub_true = tau_hat_true + qnorm(0.975)*sqrt(var_hat_true)/sqrt(n)
  coverage_true = ((ci_lb_true <= tau) & (ci_ub_true >= tau))*1
  
  # Store results
  sim_results[i, ] <- c(tau_hat_fitted, var_hat_fitted, sqrt(n)*(tau_hat_fitted - tau), 
                        ci_lb_fitted, ci_ub_fitted, coverage_fitted, 
                        tau_hat_true, var_hat_true, sqrt(n)*(tau_hat_true - tau), 
                        ci_lb_true, ci_ub_true, coverage_true)
}

# Inspect the average simulation results
apply(sim_results, 2, mean)

# Look at the histogram of sqrt(n)*(tau_hat_fitted - tau)
hist(sim_results[,3], freq = F)
# Compare to the asymptotic normal distribution implied by the 
# two variance estimators (one that accounts for estimation of pi_hat
# and another that uses the true pi_x values)
x.seq = seq(-10, 10, length.out = 10000)
lines(x.seq, dnorm(x.seq, 0, sd = sqrt(mean(sim_results[,2]))))
lines(x.seq, dnorm(x.seq, 0, sd = sqrt(mean(sim_results[,8]))), col = "blue", lty = 3)
# We observe that the "fitted weights" variance matches the asymptotic 
# distribution of sqrt(n)*(tau_hat_fitted - tau) while the "true weights" variance 
# is too conservative. We can also see this by comparing each variance estimator 
# to the asymptotic distribution of sqrt(n)*(tau_hat_true - tau)
hist(sim_results[,9], freq = F)
x.seq = seq(-15, 15, length.out = 10000)
lines(x.seq, dnorm(x.seq, 0, sd = sqrt(mean(sim_results[,2]))), col = "blue", lty = 3)
lines(x.seq, dnorm(x.seq, 0, sd = sqrt(mean(sim_results[,8]))))

# This tracks the efficiency proof of Hirano et al (2003)

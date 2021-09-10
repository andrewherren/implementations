# Lightweight univariate implementation of the slice sampler (Neal (2003))

# Define a univariate density we wish to sample
dens <- function(x) dnorm(x, -2, 2)*0.25 + dgamma(x, shape = 3, scale = 3)*0.5 + dnorm(x, 20, 2)*0.25

# Number of draws
n <- 50000

# Starting point
x0 <- 0

# Initial specifications to determine the "width" of horizontal slices
w <- 10
p <- 100
m <- 2^p

# Empty vector to store draws
x <- rep(NA, n)

for (i in 1:n){
  # Retrieve the "starting value" of x
  if (i == 1){
    x_init <- x0
  } else{
    x_init <- x[i-1]
  }
  
  ## 1. Draw a new auxiliary variable y ~ U(0, f(x_int)), 
  ##    where f is the density of random variable X
  y <- runif(1, min = 0, max = dens(x_init))
  
  ## 2. Define an interval (L, R) that contains all x* for which f(x*) >= y

  # Initialize values of L and R
  U1 <- runif(1)
  L <- x_init - w*U1
  R <- L + w
  U2 <- runif(1)
  J <- floor(m*U2)
  K <- (m - 1) - J
  
  # Iteratively reduce L
  while((J > 0) & (y < dens(L))){
    L <- L - w
    J <- J - 1
  }
  
  # Iteratively increase R
  while((K > 0) & (y < dens(R))){
    R <- R + w
    K <- K - 1
  }

  ## 2. Sample a new point in (L, R), accepting if it is in {x: f(x) > y}
  accepted <- F
  valid_draw <- F
  Lbar <- L
  Rbar <- R
  while(!accepted){
    # Move to a random point in (Lbar, Rbar)
    U <- runif(1)
    x_proposed <- Lbar + U*(Rbar - Lbar)
    
    # Check if this point is in {x: f(x) > y}
    valid_draw <- (dens(x_proposed) >= y)
    
    # Stop iterating if true
    if ((dens(x_proposed)) & (valid_draw)){
      accepted <- T
    } else{
      if (x_proposed < x_init){
        Lbar <- x_proposed
      } else{
        Rbar <- x_proposed
      }
    }
  }

  # Store the result
  x[i] <- x_proposed
}

hist(x, freq = F, breaks = 50)
x_seq <- seq(-20, 100, length.out = 100000)
lines(x_seq, dens(x_seq))

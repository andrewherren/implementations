# Several variants of linear regression
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import linprog, minimize
from numpy.linalg import inv


class LinearRegressionData:
    """
    Simple class for simulating data to test linear regression methods
    """
    def __init__(self, n, p):
        """
        Specify the number of rows (n) and number of columns (p) of the design matrix

        :param n: integer, number of rows
        :param p: integer, number of columns
        """
        self.n = n
        self.p = p
        self.X_cov = None
        self.X_mean = None
        self.weights = None
        self.epsilon = None
        self.X = None
        self.y = None

    def set_covariance(self):
        """
        Allow users to specify a covariance matrix concisely

        :return:
        """
        pass

    def set_weights(self):
        """
        Allow users to generate weights concisely

        :return:
        """
        pass

    def generate(self):
        """
        Simulate data according to user input

        :return: X, y pair of regression data
        """
        z = norm.rvs(size=(self.p, self.p))
        self.X_cov = np.matmul(np.transpose(z), z)
        self.X_mean = np.zeros(self.p)
        self.weights = np.array([5, 4, 0, 0, 0, 0, 0, 0, 0, 10])
        self.epsilon = norm.rvs(size=self.n)
        self.X = multivariate_normal.rvs(mean=self.X_mean, cov=self.X_cov, size=self.n)
        self.y = np.matmul(self.X, self.weights) + self.epsilon
        return self.X, self.y, self.weights


class OLS:
    """
    Ordinary least squares fit via matrix inversion
    """
    def __init__(self):
        self.weights = None
        self.run = False

    def fit(self, X, y):
        """
        Estimate the regression coefficients, given X and y

        :param X: design matrix
        :param y: output to be predicted
        """
        self.weights = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)),
                                 np.matmul(np.transpose(X), y))
        self.run = True


class RidgeRegression:
    """
    Ridge regression fit via QR decomposition
    """
    def __init__(self):
        self.weights = None
        self.run = False

    def augment_data(self, X, y, tau):
        """
        Prepare data for estimation by augmenting both matrices with the regularization prior

        :param X: design matrix
        :param y: output to be predicted
        :param tau: regularization hyperparameter
        :return: X_tilde, y_tilde
        """
        n, p = np.shape(X)
        lambda_prior = np.identity(p)/(np.power(tau, 2.0))
        x_tilde = np.concatenate((X, np.sqrt(lambda_prior)), axis=0)
        y_tilde = np.concatenate((y, np.zeros(p)))
        return x_tilde, y_tilde

    def fit(self, X, y, tau):
        """
        Estimate the regression coefficients, given X, y, and a tuning parameter

        :param X: design matrix
        :param y: output to be predicted
        :param tau: regularization coefficient
        """
        x_tilde, y_tilde = self.augment_data(X, y, tau)
        q, r = np.linalg.qr(x_tilde)
        self.weights = np.matmul(np.matmul(np.linalg.inv(r), q.T), y_tilde)
        self.run = True


class BayesLinearRegression:
    """
    MAP estimate for Bayesian linear regression
    """
    def __init__(self):
        self.weights = None
        self.run = False
        self.n = None
        self.p = None
        self.w0 = None
        self.v0 = None
        self.a0 = None
        self.b0 = None

    def prior_construct(self, X, y, prior="g", g=0.05):
        """
        Construct the prior parameters based on user input

        :param X: design matrix
        :param y: output to be predicted
        :param prior: (default: "g") type of prior structure to use
        """
        if prior == "g":
            self.w0 = np.zeros(p)
            # self.v0 = g * np.identity(self.p)
            self.v0 = g * np.linalg.inv(np.matmul(np.transpose(X), X))
            self.a0 = 0
            self.b0 = 0
        else:
            print("Please provide a proper prior type")

    def fit(self, X, y, prior="g", g=0.05):
        """
        Estimate the regression coefficients, given X, y, and optional prior parameters

        :param X: design matrix
        :param y: output to be predicted
        :param prior: (default: "g") type of prior structure to use
        """
        # Set up the prior
        self.n = np.shape(X)[0]
        self.p = np.shape(X)[1]
        self.prior_construct(X, y, prior=prior, g=g)

        # Compute the posterior mean / mode (same for a t distribution) on the weights
        self.vn = np.linalg.inv(np.linalg.inv(self.v0) + np.matmul(np.transpose(X), X))
        self.pmean = (np.matmul(np.linalg.inv(self.v0), self.w0) + np.matmul(np.transpose(X), y))
        self.weights = np.matmul(self.vn, self.pmean)
        self.run = True


class RobustLinearRegression:
    """
    Robust linear regression methods
    """
    def __init__(self):
        self.weights = None
        self.run = False

    def fit_laplace(self, X, y):
        """
        Provide a method for fitting a robust regression using Laplace likelihood,
        rather than Gaussian likelihood

        :param X: design matrix
        :param y: output to be predicted
        """
        n = np.shape(X)[0]
        p = np.shape(X)[1]
        empty_weights = np.zeros(p)
        pos_resid = np.zeros(n)
        neg_resid = np.zeros(n)
        x = np.concatenate((empty_weights, pos_resid, neg_resid))
        c = np.concatenate((np.zeros(p), np.ones(n), np.ones(n)))
        A = np.concatenate((X, np.identity(n), -np.identity(n)), axis=1)
        b = y
        lb = np.concatenate((np.repeat(np.NINF, p), np.repeat(0.0, 2 * n)))
        ub = np.repeat(None, p + 2 * n)
        bounds = np.array([tuple(row) for row in np.column_stack((lb, ub))])
        lp_solve = linprog(c=c, A_eq=A, b_eq=b, bounds=bounds)
        self.weights = lp_solve.x[:p]
        self.run = True

    def loss_huber(self, resid, delta):
        """
        Huber loss function of a regression residual, evaluated at a given delta

        :param resid: residuals of a regression problem (y - Xb)
        :param delta: huber loss parameter (l2 penalty for residuals
            smaller than delta, l1 penalty for residuals larger than delta
        :return: elementwise loss estimates
        """
        l2_loss = np.power(resid, 2)/2.0
        l1_loss = np.abs(resid)*delta - np.power(delta, 2)/2.0
        return np.sum(np.where(np.abs(resid) <= delta, l2_loss, l1_loss))

    def fit_huber(self, X, y, delta=0.1):
        """
        Provide a method for fitting a robust regression using Huber loss, rather than
        log likelihood

        :param X: design matrix
        :param y: output to be predicted
        :param delta:
        """
        n, p = np.shape(X)
        empty_weights = np.ones(p)
        huber_solve = minimize(fun=lambda wgt: self.loss_huber(y - np.matmul(X, wgt), delta=delta),
                               x0=empty_weights)
        self.weights = huber_solve.x
        self.run = True

    def fit(self, X, y, method="Laplace", delta=0.1):
        """
        Estimate the regression coefficients, given X and y

        :param X: design matrix
        :param y: output to be predicted
        :param method: (default: Laplace) which type of robust linear regression to implement
        """
        if method == "Laplace":
            self.fit_laplace(X, y)
        elif method == "Huber":
            self.fit_huber(X, y, delta=delta)
        else:
            print("No valid method provided")


if __name__ == "__main__":
    # Run `n_sim` simulations, applying each method to the synthetic data in each run
    n_sim = 100
    error_mat = np.zeros((n_sim, 5))

    for sim in range(n_sim):
        # Draw linear regression data
        n = 1000
        p = 10
        X, y, beta = LinearRegressionData(n, p).generate()

        # Fit an OLS regression
        ols = OLS()
        ols.fit(X, y)
        ols_weights = ols.weights
        # print("{} weights: {}".format("OLS", ols_weights))
        error_mat[sim, 0] = np.mean(np.power(ols_weights - beta, 2.0))

        # Linear regression with Laplace likelihood
        robust_reg = RobustLinearRegression()
        robust_reg.fit(X, y, method="Laplace")
        robust_weights = robust_reg.weights
        # print("{} weights: {}".format("Robust regression", robust_weights))
        error_mat[sim, 1] = np.mean(np.power(robust_weights - beta, 2.0))

        # Linear regression with Huber loss
        huber_reg = RobustLinearRegression()
        huber_reg.fit(X, y, method="Huber", delta=0.1)
        huber_weights = huber_reg.weights
        # print("{} weights: {}".format("Huber loss", huber_weights))
        error_mat[sim, 2] = np.mean(np.power(huber_weights - beta, 2.0))

        # Ridge regression
        ridge_reg = RidgeRegression()
        ridge_reg.fit(X, y, tau=100.0)
        ridge_weights = ridge_reg.weights
        # print("{} weights: {}".format("Ridge", ridge_weights))
        error_mat[sim, 3] = np.mean(np.power(ridge_weights - beta, 2.0))

        # Bayesian linear regression (unknown sigma^2, MAP estimate of betas)
        bayes_reg = BayesLinearRegression()
        bayes_reg.fit(X, y, prior="g", g=100.0)
        bayes_weights = bayes_reg.weights
        # print("{} weights: {}".format("Bayes", bayes_weights))
        error_mat[sim, 4] = np.mean(np.power(bayes_weights - beta, 2.0))

    print(np.mean(error_mat, axis=0))

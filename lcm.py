
import numpy as np
import scipy.stats as stats


class LCM:
    def __init__(self, n_class=2, tol=1e-3, max_iter=100, verbose=0, random_state=None):

        # Specify number of latent class
        self.n_class = n_class

        # Integer value: the number of iterations for EM algorithm to run
        self.max_iter = max_iter

        # Specify the tolerance value to check for convergence
        self.tol = tol

        # model parameters: initialisation
        self.beta = None
        self.theta = None
        self.poterior_prob = None
        self.log_likeld_ = [-np.inf]

        # bic estimation
        self.bic = None

        # Specify an integer value or None
        self.random_state = random_state

        # Int: either 0 or 1
        self.verbose = verbose

    def _calculate_posterior_probability(self, data):
        '''
        Using Bayes theorem (see just abve Equation 5)
        INPUTS:
            data (array): an array with shape (nxm)
        '''
        n_rows, n_cols = np.shape(data)
        posterior_prob_numerator = np.zeros(shape=(n_rows, self.n_class))
        for n in range(self.n_class):
            posterior_prob_numerator[:, n] = self.beta[n] * np.prod(stats.bernoulli.pmf(data, p=self.theta[n]), axis=1)
        posterior_prob_denominator = np.sum(posterior_prob_numerator, axis=1)
        posterior_probability = posterior_prob_numerator / np.tile(posterior_prob_denominator, (self.n_class, 1)).T
        return posterior_probability

    # Latent class model is calculated based on maximizing the log-likelihood function
    # wrt to the two parameters theta and beta using the expectation-maximization (EM) 
    # algorithm. In the maximization step, up-date the parameter estimates by maximizing 
    # the log-likelihood function given these posterior

    def _update_e(self, data):

        self.poterior_prob = self._calculate_posterior_probability(data)

    def _update_m(self, data):

        n_rows, n_cols = np.shape(data)

        # Update beta parameters
        for n in range(self.n_class):
            self.beta[n] = np.sum(self.poterior_prob[:, n]) / float(n_rows)

        # Update theta parameters
        for n in range(self.n_class):
            numerator = np.zeros((n_rows, n_cols))
            for k in range(n_rows):
                numerator[k, :] = self.poterior_prob[k, n] * data[k, :]
            numerator = np.sum(numerator, axis=0)
            denominator = np.sum(self.poterior_prob[:, n])
            self.theta[n] = numerator / denominator

        # correct numerical issues
        mask = self.theta > 1.0
        self.theta[mask] = 1.0
        mask = self.theta < 0.0
        self.theta[mask] = 0.0


    def fit(self, data):

        n_rows, n_cols = np.shape(data)

        # If number of candidates < number of class
        if n_rows < self.n_class:
            raise ValueError('''
                            LCA estimation with {n_class} components is not possible with only
                            {n_rows} samples'''.format(n_class=self.n_class, n_rows=n_rows))

        if self.verbose > 0:
            print('Maximization step with EM algorithm starting')

        self.beta = stats.dirichlet.rvs(np.ones(shape=self.n_class) / 2)[0]
        self.theta = stats.dirichlet.rvs(alpha=np.ones(shape=n_cols) / 2,
                                         size=self.n_class)

        for j in range(self.max_iter):
            if self.verbose > 0:
                print('\tEM step {n_iter}'.format(n_iter=j))

            # E-step
            self._update_e(data)

            # M-step
            self._update_m(data)

            # Check for convergence using section 2.3: Maximizing log-likelihood
            aux = np.zeros(shape=(n_rows, self.n_class))
            for n in range(self.n_class):
                # The probability that an individual i in class n
                normal_prob = np.prod(stats.bernoulli.pmf(data, p=self.theta[n]), axis=1)
                aux[:, n] = self.beta[n] * normal_prob

            #probability density function across all classes is the weighted sum
            prob_density_func = np.sum(aux, axis=1)
            log_likelihood_val = np.sum(np.log(prob_density_func))
            if np.abs(log_likelihood_val - self.log_likeld_[-1]) < self.tol:
                break
            else:
                self.log_likeld_.append(log_likelihood_val)


    def calculate_bic(self, data):
        n_rows, n_cols = np.shape(data)
        self.bic = np.log(n_rows)*(sum(self.theta.shape)+len(self.beta)) - 2.0*self.log_likeld_[-1]
        return self.bic


    def predict_proba(self, data):
        '''
        Predict the probability of class
        '''
        y_proba = self._calculate_posterior_probability(data)
        return y_proba

    def predict(self, data):
        '''
        Predict the class of the each candidate
        '''
        y_pred = np.argmax(self.predict_proba(data), axis=1)
        return y_pred


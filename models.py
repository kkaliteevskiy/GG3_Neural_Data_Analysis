import numpy as np
import numpy.random as npr
from scipy.ndimage import gaussian_filter1d as gaussian
from scipy.stats import norm



def lo_histogram(x, bins):
    """
    Left-open version of np.histogram with left-open bins covering the interval (left_edge, right_edge]
    (np.histogram does the opposite and treats bins as right-open.)
    Input & output behaviour is exactly the same as np.histogram
    """
    out = np.histogram(-x, -bins[::-1])
    return out[0][::-1], out[1:]


def gamma_isi_point_process(rate, shape):
    """
    Simulates (1 trial of) a sub-poisson point process (with underdispersed inter-spike intervals relative to Poisson)
    :param rate: time-series giving the mean spike count (firing rate * dt) in different time bins (= time steps)
    :param shape: shape parameter of the gamma distribution of ISI's
    :return: vector of spike counts with same shape as "rate".
    """
    sum_r_t = np.hstack((0, np.cumsum(rate)))
    gs = np.zeros(2)
    while gs[-1] < sum_r_t[-1]:
        gs = np.cumsum( npr.gamma(shape, 1 / shape, size=(2 + int(2 * sum_r_t[-1]),)) )
    y, _ = lo_histogram(gs, sum_r_t)

    return y



class StepModel():
    """
    Simulator of the Stepping Model of Latimer et al. Science 2015.
    """
    def __init__(self, m=50, r=10, x0=0.2, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        """
        Simulator of the Stepping Model of Latimer et al. Science 2015.
        :param m: mean jump time (in # of time-steps). This is the mean parameter of the Negative Binomial distribution
                  of jump (stepping) time
        :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time
                  (Note that it is more customary to parametrise the NB distribution by its parameter p and r,
                  instead of m and r, where p is so-called "probability of success" (see Wikipedia). The two
                  parametrisations are equivalent and one can go back-and-forth via: m = r (1-p)/p and p = r / (m + r).)
        :param x0: determines the pre-jump firing rate, via  R_pre = x0 * Rh (see below for Rh)
        :param Rh: firing rate of the "up" state (the same as the post-jump state in most of the project tasks)
        :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                            see https://en.wikipedia.org/wiki/Gamma_distribution
        :param Rl: firing rate of the post-jump "down" state (rarely used)
        :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
        """
        self.modelel_type = 'Step'
        self.m = m
        self.r = r
        self.x0 = x0

        self.p = r / (m + r)

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt


    @property
    def params(self):
        return self.m, self.r, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl


    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                 trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y


    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
    rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        ts = np.arange(T)

        spikes, jumps, rates = [], [], []
        for tr in range(Ntrials):
            # sample jump time
            jump = npr.negative_binomial(self.r, self.p)
            jumps.append(jump)

            # first set rate at all times to pre-step rate
            rate = np.ones(T) * self.x0 * self.Rh
            # then set rates after jump to self.Rh
            rate[ts >= jump] = self.Rh
            rates.append(rate)

            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates)
        else:
            return np.array(spikes), np.array(jumps)


class RampModel():
    """
    Simulator of the Ramping Model (aka Drift-Diffusion Model) of Latimer et al., Science (2015).
    """
    def __init__(self, beta=0.5, sigma=0.2, x0=.2, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        """
        Simulator of the Ramping Model of Latimer et al. Science 2015.
        :param beta: drift rate of the drift-diffusion process
        :param sigma: diffusion strength of the drift-diffusion process.
        :param x0: average initial value of latent variable x[0]
        :param Rh: the maximal firing rate obtained when x_t reaches 1 (corresponding to the same as the post-step
                   state in most of the project tasks)
        :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                            see https://en.wikipedia.org/wiki/Gamma_distribution
        :param Rl: Not implemented. Ignore.
        :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
        """
        self.beta = beta
        self.sigma = sigma
        self.x0 = x0

        self.model_type = 'Ramp'

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt


    @property
    def params(self):
        return self.mu, self.sigma, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl


    def f_io(self, xs, b=None):
        if b is None:
            return self.Rh * np.maximum(0, xs)
        else:
            return self.Rh * b * np.log(1 + np.exp(xs / b))


    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                 trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y


    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

       # simulate all trials in parallel (using numpy arrays and broadcasting)

        # first, directly integrate/sum the drift-diffusion updates
        # x[t+1] = x[t] + β dt + σ √dt * randn (with initial condition x[0] = x0 + σ √dt * randn)
        # to get xs in shape (Ntrials, T):
        ts = np.arange(T)
        xs = self.x0 + self.beta * dt * ts + self.sigma * np.sqrt(dt) * np.cumsum(npr.randn(Ntrials, T), axis=1)
        # in each trial set x to 1 after 1st passage through 1; padding xs w 1 assures passage does happen, possibly at T+1
        taus = np.argmax(np.hstack((xs, np.ones((xs.shape[0],1)))) >= 1., axis=-1)
        xs = np.where(ts[None,:] >= taus[:,None], 1., xs)
        # # the above 2 lines are equivalent to:
        # for x in xs:
        #     if np.sum(x >= 1) > 0:
        #         tau = np.nonzero(x >= 1)[0][0]
        #         x[tau:] = 1

        rates = self.f_io(xs) # shape = (Ntrials, T)

        spikes = np.array([self.emit(rate) for rate in rates]) # shape = (Ntrial, T)

        spikes = np.array(spikes)
        xs = np.array(xs)
        rates = np.array(rates)

        transition_time = self.get_transition_time(xs)
        
        if get_rate:
            return spikes, xs, rates#, transition_time
        else:
            return spikes, xs
        
    # def get_transition_time(self, xs):
    #     true_transitions = []

    # for x in xs_step:
    #     for i in range(len(x)):
    #         if x[i] == shmm.r:
    #             true_transitions.append(i)
    #             break



class HMM_Step_Model():
    def __init__(self, m = 32, r = 6, x0 = 0.2, Rh = 75, T = 100, isi_gamma_shape=None):

        self.modelel_type = 'Step'
        self.isi_gamma_shape = isi_gamma_shape
        self.m = m
        self.r = r
        self.x0 = x0
        self.p = r / (m + r)
        self.Rh = Rh
        self.T = T
        self.dt = 1/T
        self.P = self.get_transition_matrix()
        self.pi0 = np.zeros(r)
        self.pi0[0] = 1
        self.lambdas = self.get_rate(np.arange(self.r)) * self.dt

    
    def get_transition_matrix(self):
        transition_matrix = np.zeros([self.r,self.r])
        for i in range(self.r - 1):
            transition_matrix[i][i] = 1 - self.p
            transition_matrix[i][i+1] = self.p
        transition_matrix[self.r-1][self.r-1] = 1
        return transition_matrix


    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                    trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)
        return y
    
    def simulate_states(self):
        '''returns a sequence of states for a single trial of length T'''
        x = [0]
        p = self.p 
        for t in range(1, self.T + self.r - 1):
            if x[t-1] == self.r-1:
                x.append(self.r-1)
            else:
                x.append(x[t-1] + np.random.choice([0,1], p = [1-p, p]))
        return x[-self.T:]

    def get_rate(self, x):
        rate = np.ones(len(x)) * self.x0 * self.Rh
        rate[x == (self.r-1)] = self.Rh
        return rate
    
    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.

        spikes, xs, rates = [], [], []

   
        for i in range(0, Ntrials):
            x = self.simulate_states()
            xs.append(x) 
            rate = self.get_rate(np.array(x))
            rates.append(rate)

            spikes.append(self.emit(rate))
        
        spikes = np.array(spikes)
        xs = np.array(xs)
        rates = np.array(rates)

        if spikes.shape[0] == 1:
            spikes = np.squeeze(spikes, axis=0)
        if xs.shape[0] == 1:
            xs = np.squeeze(xs, axis=0)
        if rates.shape[0] == 1:
            rates = np.squeeze(rates, axis=0)
        
        if get_rate:
            return spikes, xs, rates
        else:
            return spikes, xs
        

class HMM_Ramp_Model():

    def __init__(self, beta = 2, sigma = 2, x0 = 0.2, Rh = 75, K = 100, T = 100, isi_gamma_shape = None):
        self.T = T
        self.dt = 1/T
        self.isi_gamma_shape = isi_gamma_shape
        self.model_type = 'Ramp'
        self.beta = beta
        self.sigma = sigma
        self.K = K
        self.x0 = x0
        self.Rh = Rh
        self.P = self.get_transition_matrix()
        self.pi0 = None
        self.set_initial_distribution()
        self.lambdas = self.get_rate(np.linspace(0,1,num = self.K)) * self.dt

    def set_initial_distribution(self):
        pi0 = self.normal_dist(mean = self.x0, sd = self.sigma * np.sqrt(self.dt))
        self.pi0 = pi0
                                                      
    def simulate_states(self):
        '''returns a sequence of states for a single trial of length T'''
        s = []
        s.append(np.random.choice(np.arange(self.K), p=self.pi0))
        for t in range(1, self.T):
            s.append(np.random.choice(np.arange(self.K), p=self.P[s[t-1]]))
        x = np.array(s) / (self.K - 1)
        return x

    def get_transition_matrix(self):
        trans = np.empty([self.K,self.K])
        arr = np.linspace(0,1,num = self.K)
        for i, x in enumerate(arr):
            trans[i] = self.normal_dist(mean = x + self.beta * self.dt, sd = self.sigma * np.sqrt(self.dt))
        trans[self.K-1] = np.zeros(self.K)
        trans[self.K-1][self.K-1] = 1
        return trans
    
    def normal_dist(self, mean = 0, sd = 1):
        x = np.linspace(0, 1, self.K)
        step = (x[1] - x[0]) * np.ones(self.K)
        kernel = norm.cdf(x + step/2, loc=mean, scale=sd) - norm.cdf(x-step/2, loc=mean, scale=sd)
        kernel = kernel / np.sum(kernel)
        # prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        return kernel


    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                    trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)
        return y
    
    def get_rate(self, xs):
        rate = self.Rh * np.maximum(0, xs)
        return rate

    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.T = T
        self.dt = dt

        spikes, xs, rates = [], [], []

        for _ in range(Ntrials):
            x = self.simulate_states()
            xs.append(x)
            rate = self.get_rate(np.array(x))
            rates.append(rate)

            spikes.append(self.emit(rate))
        
        spikes = np.array(spikes)
        xs = np.array(xs)
        rates = np.array(rates)
        if spikes.shape[0] == 1:
            spikes = np.squeeze(spikes, axis=0)
        if xs.shape[0] == 1:
            xs = np.squeeze(xs, axis=0)
        if rates.shape[0] == 1:
            rates = np.squeeze(rates, axis=0)

        if get_rate:
            return np.array(spikes), np.array(xs), np.array(rates)
        else:
            return np.array(spikes), np.array(xs)

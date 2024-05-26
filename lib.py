import models
import numpy as np
import matplotlib.pyplot as plt 

def plot_latent_variable(model, xs):
    '''plot the latent variable only works for Ramp model
    :param model: model object
    :param xs: latent variable time-series x_t in trial j'''
    if model.model_type == "Ramp":
        plt.figure(figsize=(6, 3))
        plt.grid(False)
        plt.plot(xs.T)
        plt.ylabel('$x_t$')
        plt.xlabel('Time index, $\Delta t={dt}$'.format(dt=model.dt))
    elif model.model_type == "Step":
        pass

def plot_jump_time_histogram(transitions):
    plt.figure(figsize=(6, 3))
    plt.hist([t for t in transitions if t is not None], bins=20)
    plt.xlabel('Transition Time')
    plt.ylabel('Frequency')
    plt.title('Transition Time Histogram')
    plt.grid(True)


def get_quadratic_fit(x, y):
    '''fit a quadratic function to the data
    :param x: x values
    :param y: y values
    :return: coefficients of the quadratic fit'''
    A = np.vstack([x**2, x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]
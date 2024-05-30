
#%%
import models
import numpy as np
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt


N_realizations = 10000
T = 100


#%% optimal setting
m = 32.82906699667256
r = 6.193394287810123
beta = 1.474548818691456
sigma = 0.32658335633379126

#%% NEW BEST
m = 30.12872134023911
r = 2.434064053799087
beta = 1.4683905859999828
sigma = 0.2691943299818452

#%%initial guess
m = 29
r = 7.6
beta = 1.12
sigma = 0.5




#%%
t = np.arange(0, 1, 0.01)
plt.plot(t, f_ramp_estimate(beta, sigma)*100, label = 'ramp')
# plt.plot(f_step_estimate(m-5,r-3.3))
plt.plot(t, f_step_estimate(m,r)*100, label = 'step')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
plt.legend()

#%%
def f_step_estimate(m, r):
    step_model = models.StepModel(m=m, r=r, x0=0.1, Rh=50)
    spikes, xs, rates = step_model.simulate(N_realizations, T=T)
    avg_rate = np.mean(spikes, axis=0)
    return avg_rate

def f_ramp_estimate(beta, sigma):
    ramp_model = models.RampModel(beta, sigma, x0=0.1, Rh=50)
    spikes, xs, rates = ramp_model.simulate(N_realizations, T=T)
    avg_rate = np.mean(spikes, axis=0)
    return avg_rate

def SE(x_0):
    m,r,beta,sigma = x_0
    print(f"m = {m}, r = {r}, beta = {beta}, sigma = {sigma}")
    se = np.sum((f_ramp_estimate(beta, sigma) - f_step_estimate(m, r))**2)
    print(f"SE = {se}")
    return se

# def callback(x):
#     print(f"m = {m}, r = {r}, beta = {beta}, sigma = {sigma}")
#     print(f"MSE = {MSE(x)}")
# %%

x0 = [m, r, beta, sigma] #initial guess
result = minimize(SE, x0, method = 'Nelder-Mead', tol=1e-5)
x_optimal = result.x
print(f"Optimal values: {x_optimal}")

# %%

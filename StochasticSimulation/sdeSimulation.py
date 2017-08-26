from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
num_paths = 100
grid_points = 100
T = 1.
y0 = 1.

def em_iteration(y_t, dW, drift, diffusion, dt):
    return y_t + drift(y_t) * dt + diffusion(y_t) * dW

drift = lambda x: x
diffusion = lambda x: x

def simulate_sde_em(drift, diffusion, iteration_method, num_paths, grid_points, T=1, y0=1):
    h = T / grid_points
    t = np.linspace(0, T, grid_points + 1)

    em_iteration = lambda prev, x: iteration_method(prev, x, drift, diffusion, h)

    # Wiener Path realizations
    dW = np.random.randn(num_paths, grid_points) * np.sqrt(h)
    W = np.hstack((np.zeros((num_paths, 1)), np.cumsum(dW, axis=1)))

    # apply the iteration on each Wiener path realization, I am using the python function accumulate to implement the iteration.
    em_paths = np.array([list(accumulate(np.append([y0], p), em_iteration)) for p in dW])

    return em_paths, W, t

# Compute paths and the exact simulation
em_paths_to_plot, W, t_to_plot = simulate_sde_em(drift, diffusion, em_iteration, num_paths, grid_points, T, y0)
exact_paths_to_plot = y0 * np.exp(0.5 * t_to_plot + W)

# Compute the errs vs. Time step

N = 10**np.arange(5)
errs = []
time_step = []

p_thresh = np.linspace(0, 0.9, 10)
errs_cond_percentile = []

for n in N:
    em_paths, W, t = simulate_sde_em(drift, diffusion, em_iteration, num_paths, n, T, y0)
    exact_paths = y0 * np.exp(0.5*t + W)
    # compute path errors
    err = np.mean(np.sqrt(np.mean((em_paths - exact_paths)**2, axis=1)))
    errs.append(err)
    time_step.append(T/n)
    # compute path errors for each percentile thresh
    perc_err = []
    for p in p_thresh:
        terminal = em_paths[:, -1]
        thresh = np.percentile(terminal, p*100)
        err = np.mean(np.sqrt(np.mean((em_paths[terminal>=thresh] - exact_paths[terminal>=thresh])**2, axis=1)))
        perc_err.append(err)
    errs_cond_percentile.append(perc_err)

# Plot log(paths errors) vs. log(time step)
plt.figure(figsize=(20, 10))

# ----------------------------------------------------------------------------------------------
# Let's plot the sample paths to see the difference between the exact solution and EM approximation.
# ----------------------------------------------------------------------------------------------
plt.subplot(221)
plt.plot(em_paths_to_plot.T, linestyle='--')
plt.plot(em_paths_to_plot.mean(axis=0), color='black', lw=2, linestyle='-', label='cross-sectional mean')
plt.title('Sample paths from the iteration scheme')
plt.legend(loc=0)

plt.subplot(222)
colormap = plt.cm.gist_ncar
colors = [colormap(i) for i in np.linspace(0, 0.9, 5)]

for i, p_i in enumerate(np.random.randint(0, num_paths - 1, 5)):
    _ = plt.plot(em_paths_to_plot[p_i], color=colors[i], linestyle='-')
    _ = plt.plot(exact_paths_to_plot[p_i], color=colors[i], linestyle='-.', lw=2)
plt.title('Comparison of the exact solution to the approximation (- Approx, -. Exact)')

fit = np.polyfit(np.log(time_step),np.log(errs),1)
fit_fn = np.poly1d(fit)

plt.subplot(223)
plt.plot(np.log(time_step), np.log(errs), 'b-')
eqn = '$ y=%sx %s %s $' % (np.round(fit[0], 3), '-' if np.sign(np.round(fit[1], 3)) == -1 else "+", np.abs(np.round(fit[1], 3)))
plt.plot(np.log(time_step), fit_fn(np.log(time_step)), 'k--', label=eqn)
plt.ylabel('$log( \ E \ (\mid {y_t - \hat{y}_t} \mid)$')
plt.xlabel('$log(\Delta T)$')
plt.title('Path errors vs. Time step')
plt.legend()

# Plot
plt.subplot(224)
plt.plot(p_thresh, np.log(errs_cond_percentile[0]), label='N: %s' % N[0])
plt.plot(p_thresh, np.log(errs_cond_percentile[1]), label='N: %s' % N[1])
plt.plot(p_thresh, np.log(errs_cond_percentile[2]), label='N: %s' % N[2])
plt.ylabel('$log( \ E \ ((\mid {y_t - \hat{y}_t} \mid) )$')
plt.xlabel('percentile')
plt.title('Conditional Path error vs. percentile of terminal distribution')
plt.legend(loc=0)

plt.tight_layout(h_pad=3, w_pad=3)

plt.show()
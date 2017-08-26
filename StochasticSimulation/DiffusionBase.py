from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import inspect
from types import SimpleNamespace
# from sympy import *
import sympy


class DiffusionBase:
    def __init__(self, drift, diffusion, approximation_func, seed=100):
        self.drift = drift
        self.diffusion = diffusion
        self.d_diffusion = self.get_derivatives(self.diffusion)
        self.iteration_method = approximation_func
        self.rng = np.random.RandomState(seed)

    def simulate_paths(self, num_paths, path_length, T=1, y0=1):
        h = T / path_length
        t = np.linspace(0, T, path_length + 1)

        iteration_method = lambda prev, x: self.iteration_method(prev, x, h, self)

        # Wiener Path realizations
        dW = self.rng.randn(num_paths, path_length) * np.sqrt(h)
        W = np.hstack((np.zeros((num_paths, 1)), np.cumsum(dW, axis=1)))

        # apply the iteration on each Wiener path realization, I am using the python function accumulate to implement the iteration.
        em_paths = np.array([list(accumulate(np.append([y0], p), iteration_method)) for p in dW])

        return em_paths, W, t

    def get_derivatives(self, func):
        arg_symbols = sympy.symbols(list(inspect.signature(drift).parameters.keys()))
        sym_func = func(*arg_symbols)

        return [sympy.lambdify(arg_symbols, sym_func.diff(a)) for a in arg_symbols][0]

    def path_error(self, exact_solution, log_10_max_grid_points=5):
        N = 10 ** np.arange(log_10_max_grid_points)
        errs = []
        time_step = []

        p_thresh = np.linspace(0, 0.9, 10)
        errs_cond_percentile = []

        num_paths = 1000
        T = 1
        y0 = 1

        for n in N:
            # print(' --- computing errors for grid points %s' % n)
            em_paths, W, t = self.simulate_paths(num_paths, n, T, y0)
            exact_paths = exact_solution(y0, W, t)
            # compute path errors
            err = np.mean(np.sqrt(np.mean((em_paths - exact_paths) ** 2, axis=1)))
            errs.append(err)
            time_step.append(T / n)
            # compute path errors for each percentile thresh
            perc_err = []
            for p in p_thresh:
                terminal = em_paths[:, -1]
                thresh = np.percentile(terminal, p * 100)
                err = np.mean(
                    np.sqrt(np.mean((em_paths[terminal >= thresh] - exact_paths[terminal >= thresh]) ** 2, axis=1)))
                perc_err.append(err)
            errs_cond_percentile.append(perc_err)
        return {'path_errors': errs, 'time_step':time_step, 'percentile_levels':p_thresh, 'conditional_errors':errs_cond_percentile}

    def plot_path_and_error(self, exact_solution, plot=True, log10_max_grid_points=5):
        num_paths = 100
        y0=1
        T=1

        em_paths_to_plot, W, t = self.simulate_paths(num_paths, 100, T, y0)
        exact_paths_to_plot = exact_solution(1, W, t)

        res = self.path_error(exact_solution, log_10_max_grid_points=log10_max_grid_points)
        time_step = res['time_step']
        errs = res['path_errors']
        p_thresh = res['percentile_levels']
        errs_cond_percentile = res['conditional_errors']

        self.temp_results = SimpleNamespace(y0=y0, T=T, num_paths=num_paths, approx_paths=em_paths_to_plot,
                                            W=W, t=t, exact_paths=exact_paths_to_plot,
                                            errs={'time_step':time_step, 'errs':errs},
                                            cond_errs={'percentile':p_thresh, 'conditional_errors':errs_cond_percentile})
        # Plot log(paths errors) vs. log(time step)
        if plot:
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

            for i, p_i in enumerate(self.rng.randint(0, num_paths - 1, 5)):
                _ = plt.plot(em_paths_to_plot[p_i], color=colors[i], linestyle='-')
                _ = plt.plot(exact_paths_to_plot[p_i], color=colors[i], linestyle='-.', lw=2)
            plt.title('Comparison of the exact solution to the approximation (- Approx, -. Exact)')

            fit = np.polyfit(np.log(time_step), np.log(errs), 1)
            fit_fn = np.poly1d(fit)

            plt.subplot(223)
            plt.plot(np.log(time_step), np.log(errs), 'b-')
            eqn = '$ y=%sx %s %s $' % (
            np.round(fit[0], 3), '-' if np.sign(np.round(fit[1], 3)) == -1 else "+", np.abs(np.round(fit[1], 3)))
            plt.plot(np.log(time_step), fit_fn(np.log(time_step)), 'k--', label=eqn)
            plt.ylabel('$log( \ E \ (\mid {y_t - \hat{y}_t} \mid)$')
            plt.xlabel('$log(\Delta T)$')
            plt.title('Path errors vs. Time step')
            plt.legend()

            # Plot
            plt.subplot(224)
            plt.plot(p_thresh, np.log(errs_cond_percentile[0]), label='time_step: %s' % time_step[0])
            plt.plot(p_thresh, np.log(errs_cond_percentile[1]), label='time_step: %s' % time_step[1])
            plt.plot(p_thresh, np.log(errs_cond_percentile[2]), label='time_step: %s' % time_step[2])
            plt.ylabel('$log( \ E \ ((\mid {y_t - \hat{y}_t} \mid) )$')
            plt.xlabel('percentile')
            plt.title('Conditional Path error vs. percentile of terminal distribution')
            plt.legend(loc=0)

            plt.tight_layout(h_pad=3, w_pad=3)
            plt.show()



if __name__ == '__main__':
    drift = lambda x: x
    diffusion = lambda x: x
    approx_func = lambda y_t, dW, dt, obj: y_t + obj.drift(y_t) * dt + obj.diffusion(y_t) * dW
    exact_solution = lambda y0, W, t: y0 * np.exp(0.5*t + W)

    objEuler = DiffusionBase(drift, diffusion, approx_func)
    objEuler.plot_path_and_error(exact_solution,plot=False)
    euler_results = objEuler.temp_results

    drift = lambda x: x
    diffusion = lambda x: x
    advance = lambda y_t, dW, dt, obj: y_t + obj.drift(y_t) * dt + obj.diffusion(y_t) * dW + \
                                       obj.diffusion(y_t) * obj.d_diffusion(y_t) * 0.5 * (dW ** 2 - dt)
    objMilstein = DiffusionBase(drift, diffusion, advance)  # Boiler plate object for all the analysis
    objMilstein.plot_path_and_error(exact_solution, plot=False)
    milstein_results = objMilstein.temp_results

    print('Check all the W are same in euler and milstein: %s' % np.all(np.ravel(np.round(euler_results.W - milstein_results.W, 14))==0))
    print('Check all the W are same in euler and milstein: %s' % np.all(np.ravel(np.round(euler_results.exact_paths - milstein_results.exact_paths, 14)) == 0))

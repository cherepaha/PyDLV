import pandas as pd
import numpy as np
import scipy as sp

class DLGenerator:
    '''
    This class is responsible for generating decision landScape parameters using the experimental
    trajectories. The two methods, fit_ds_single_traj and fit_ds_mult_traj, provide external 
    interface to the core routine, dlg_minimize, which in turn wraps around several optimisation 
    algorithms available in scipy.opimize. The method get_model_dl is used for getting 
    visualisation-ready data.
    '''
    n_cells = 25
    
    def __init__(self, model, x_lim = [-1.1, 1.1], y_lim = [0.0, 1.1]):
        self.model = model
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.baseline_coeffs = self.model.get_baseline_params()
        self.param_boundaries = self.model.get_parameter_bounds()
        self.param_names = self.model.get_param_names()        

    def fit_dl_single_traj(self, trajectory, method=9):
        '''
        trajectory: long dataframe containing t, x, y, vx, vy for each time step       
        '''
        f = lambda coeffs: self.model.model_error(coeffs, trajectory)
        f_jac = lambda coeffs: self.model.model_error_jac(coeffs, trajectory)
            
        return self.dlg_minimize(f, f_jac, method)
    
    def fit_dl_mult_traj(self, trajectories, method=9, groupby_cols=['trial_no']):
        '''
        trajectories: long dataframe containing t, x, y, vx, vy for each time step 
        for each trajectory
        '''
        f = lambda coeffs: self.model.model_error_multiple_traj(coeffs, trajectories, groupby_cols)
        f_jac = lambda coeffs: self.model.model_error_jac_multiple_traj(coeffs, trajectories, 
                                                                        groupby_cols)
                    
        return self.dlg_minimize(f, f_jac, method)
    
    def dlg_minimize(self, f, f_jac=[], method=9):
        '''
        Wrapper of scipy.optimize algorithms. Accepts cost function and its jacobian and returns 
        the dataframe with optimization results. Jacobian is required for methods 3 to 9, and 12.
        Recommended methods are L-BFGS-B (6) and SLSQP (9)
        '''
        if (method == 1):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='Powell')        
        elif (method == 2):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='Nelder-Mead')        
        elif (method == 3):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='CG', jac=f_jac)
        elif (method == 4):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='BFGS', jac=f_jac)        
        elif (method == 5):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='Newton-CG', 
                                              jac=f_jac)
        elif (method == 6):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='L-BFGS-B',
                                              bounds=self.param_boundaries, jac=f_jac)
        elif (method == 7):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='TNC',
                                              bounds=self.param_boundaries, jac=f_jac)
        elif (method == 8):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='COBYLA',
                                              bounds=self.param_boundaries, jac=f_jac)
        elif (method == 9):
            fit_coeffs = sp.optimize.minimize(f, self.baseline_coeffs, method='SLSQP',
                                              bounds=self.param_boundaries, jac=f_jac)
        elif (method == 10):
            fit_coeffs = sp.optimize.differential_evolution(f, polish = True, mutation=(1.0, 1.5), 
                                                            bounds=self.param_boundaries)                
        elif (method == 11):
            minimizer_kwargs = {'method': 'Powell'}
            fit_coeffs = sp.optimize.basinhopping(f, self.baseline_coeffs, niter=200,
                                                  minimizer_kwargs=minimizer_kwargs)
        elif (method == 12):
            minimizer_kwargs = {'method': 'CG', 'jac': True}
            f_with_jac = lambda coeffs: [f(coeffs), f_jac(coeffs)]
            fit_coeffs = sp.optimize.basinhopping(f_with_jac, self.baseline_coeffs, niter=200,
                                                  minimizer_kwargs=minimizer_kwargs)
                                           
        opt_results = np.array((fit_coeffs.fun, fit_coeffs.nfev))   
        opt_results = np.concatenate((opt_results, fit_coeffs.x))
        opt_results = pd.DataFrame(opt_results).transpose()
        opt_results.columns= ['error', 'nfev'] + self.param_names
        return opt_results
    
    def get_model_dl(self, params):
        x_grid = np.linspace(self.x_lim[0], self.x_lim[1], self.n_cells+1)
        y_grid = np.linspace(self.y_lim[0], self.y_lim[1], self.n_cells+1)
        X, Y = np.meshgrid((x_grid[:-1]+x_grid[1:])/2.0, (y_grid[:-1]+y_grid[1:])/2.0, 
                           indexing='xy')
        dl = self.model.V(X, Y, params)/params[0]
        
        return x_grid, y_grid, dl

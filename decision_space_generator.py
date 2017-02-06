import pandas as pd
import numpy as np
from scipy import integrate
from scipy.interpolate import interp2d
import scipy as sp

class DecisionSpaceGenerator:
#    absolute coordinates
#    x_lim = [-275, 275]
#    y_lim = [-50, 375]
    
#    relative coordinates
    n_cells = 25       
    x_lim = [-1.3, 1.3]
    y_lim = [0.0, 1.3]

    def fit_ds_single_traj(self, trajectory, model, mode=1):
        # see fit_log.xlsx for the list of values for mode parameter
        # trajectory: dataframe containing t, x, y, vx, vy for each time step
        print(trajectory.iloc[1].name)
        
        f = lambda coeffs: model.model_error(coeffs, trajectory)
        f_jac = lambda coeffs: model.model_error_jac(coeffs, trajectory)
        coeffs_guess = model.get_parameter_guess()
        param_boundaries = model.get_parameter_bounds()
        param_names = model.get_param_names()
            
        return self.dsg_minimize(f, f_jac, coeffs_guess, param_names, param_boundaries, mode)
    
    def fit_ds_mult_traj(self, trajectories, model, mode=1):
#        print(trajectories.iloc[1].name[0])
        
        f = lambda coeffs: model.model_error_multiple_traj(coeffs, trajectories)
        f_jac = lambda coeffs: model.model_error_jac_multiple_traj(coeffs, trajectories)
        coeffs_guess = model.get_parameter_guess()
        param_boundaries = model.get_parameter_bounds()
        param_names = model.get_param_names()
                    
        return self.dsg_minimize(f, f_jac, coeffs_guess, param_names, param_boundaries, mode)
    
    def dsg_minimize(self, f, f_jac, coeffs_guess, param_names, param_boundaries, mode = 1):
        if (mode == 1):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='Powell')        

        if (mode == 2):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='Nelder-Mead')        

        if (mode == 3):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='CG', jac=f_jac)

        if (mode == 4):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='BFGS', jac=f_jac)
        
        if (mode == 5):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='Newton-CG', jac=f_jac)

        if (mode == 6):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='L-BFGS-B',
                                          bounds = param_boundaries, jac=f_jac)

        if (mode == 7):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='TNC',
                                          bounds = param_boundaries, jac=f_jac)
                                          
        if (mode == 8):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='COBYLA',
                                          bounds = param_boundaries, jac=f_jac)
        
        if (mode == 9):
            fit_coeffs = sp.optimize.minimize(f, coeffs_guess, method='SLSQP',
                                          bounds = param_boundaries, jac=f_jac)

        if (mode == 10):
            fit_coeffs = sp.optimize.differential_evolution(f, polish = True, 
                                                         mutation = (1.0, 1.5),
                                                         bounds = param_boundaries)                
        if (mode == 11):
            minimizer_kwargs = {'method': 'Powell'}
            fit_coeffs = sp.optimize.basinhopping(f, coeffs_guess, 
                                           minimizer_kwargs=minimizer_kwargs, niter=200)
        if (mode == 12):
            minimizer_kwargs = {'method': 'CG', 'jac': True}
            f_with_jac = lambda coeffs: [f(coeffs), f_jac(coeffs)]
            fit_coeffs = sp.optimize.basinhopping(f_with_jac, coeffs_guess, 
                                           minimizer_kwargs=minimizer_kwargs, niter=200)
                                           
        opt_results = np.array((fit_coeffs.fun, fit_coeffs.nfev))   
        opt_results = np.concatenate((opt_results, fit_coeffs.x))
        opt_results = pd.DataFrame(opt_results).transpose()
        opt_results.columns= ['error', 'nfev'] + param_names
        return opt_results
    
    def get_model_ds(self, model, coeffs):         
        x_grid = np.linspace(self.x_lim[0], self.x_lim[1], self.n_cells+1)
        y_grid = np.linspace(self.y_lim[0], self.y_lim[1], self.n_cells+1)
        
        X, Y = np.meshgrid((x_grid[:-1]+x_grid[1:])/2.0, (y_grid[:-1]+y_grid[1:])/2.0, 
                           indexing='xy')        
#        X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')
        ds = model.V(X, Y, coeffs)/coeffs[0]
        
        return x_grid, y_grid, ds

    def get_model_vector_field(self, model, coeffs):
        x_grid = np.linspace(self.x_lim[0], self.x_lim[1], self.n_cells+1)
        y_grid = np.linspace(self.y_lim[0], self.y_lim[1], self.n_cells+1)
        
        X, Y = np.meshgrid((x_grid[:-1]+x_grid[1:])/2.0, (y_grid[:-1]+y_grid[1:])/2.0, 
                           indexing='xy')
        f = model.model([X, Y], 0, coeffs)
                
        return f[0], f[1], X, Y
    
    def get_model_ds_int(self, model, coeffs, return_rhs=False):
        n_cells = 25       
        x_lim = [-1.2, 1.2]
        y_lim = [-0.1, 1.2]
         
        x_grid = np.linspace(x_lim[0], x_lim[1], n_cells+1)
        y_grid = np.linspace(y_lim[0], y_lim[1], n_cells+1)
         
        f_x = lambda x, y: model.dV_dx(x, y, coeffs)
        f_y = lambda x, y: model.dV_dy(x, y, coeffs)
        
        if return_rhs:
            xx, yy = np.meshgrid(x_grid, y_grid)
            return -f_x(xx, yy), -f_y(xx, yy), xx, yy
             
        ds = np.zeros((n_cells, n_cells))        
        # for each x,y on the grid calculate the line integral of f_x and f_y 
        # along the straight line between (0,0) to (x,y)
        for i in range(n_cells):
            for j in range(n_cells):
                # parametrize 2-d functions f_x and f_y so that f_x_s and f_y_s are 1-d functions
                # s=0 corresponds to (0,0) and s=1 to (x_grid[i], y_grid[j])
                f_x_s = lambda s: f_x(s*x_grid[i], s*y_grid[j])
                f_y_s = lambda s: f_y(s*x_grid[i], s*y_grid[j])
                integrand = lambda s: np.dot([f_x_s(s), f_y_s(s)], [x_grid[i], y_grid[j]])
                ds[j, i], _ = integrate.quad(integrand, 0, 1)
         
        return ds, x_grid, y_grid
    
    def generate_ds_control(self, model, coeffs, return_rhs=False):
        n_cells = 25       
        x_lim = [-1.2, 1.2]
        y_lim = [-0.5, 1.5]
         
        x_grid = np.linspace(x_lim[0], x_lim[1], n_cells+1)
        y_grid = np.linspace(y_lim[0], y_lim[1], n_cells+1)
        
        
        f_x = lambda x, y: model.f_x_control(coeffs,x,y)
        f_y = lambda x, y: model.dV_dy(x, y, coeffs)
                            
        if return_rhs:
            xx,yy=np.meshgrid(x_grid,y_grid)
            return -f_x(xx,yy),-f_y(xx,yy),xx,yy
    
        ds = np.zeros((n_cells, n_cells))        
        for i in range(n_cells):
            for j in range(n_cells):
                f_x_s = lambda s: f_x(s*x_grid[i], s*y_grid[j])
                f_y_s = lambda s: f_y(s*x_grid[i], s*y_grid[j])
                integrand = lambda s: np.dot([f_x_s(s), f_y_s(s)], [x_grid[i], y_grid[j]])
                ds[j, i], _ = integrate.quad(integrand, 0, 1)
         
        return ds, x_grid, y_grid
        
    def generate_ds(self, data, n_cells=25, gamma=1.0, return_u_xy=False):
        # generate decision space from the data containing x, y, v_x, v_y, a_x, a_y time series
        # using interval-wise averaging (as in ohora2013local), i.e., 
        # assuming that motion in x and y directions are independent: u = u_x + u_y
        # gamma is the parameter defining contribution of acceleration
    
        x_grid, x_step = np.linspace(self.x_lim[0], self.x_lim[1], n_cells+1, retstep=True)
        y_grid, y_step = np.linspace(self.y_lim[0], self.y_lim[1], n_cells+1, retstep=True)

        # assuming that x- and y-grids are the same size, mean_derivatives is filled by 
        # vx, ax, vy, ay averaged over each interval along either axis
        mean_derivatives = np.zeros((n_cells, 4))
        
        for i in range(n_cells):
            [x_left, x_right] = [x_grid[i], x_grid[i+1]]
            [y_left, y_right] = [y_grid[i], y_grid[i+1]]

            mean_derivatives[i, :2] = data.loc[(data.x >= x_left) & (data.x < x_right)]. \
                                    groupby(level=['subj_id', 'trial_no']).mean(). \
                                    loc[:, ['vx','ax']].mean().values
            mean_derivatives[i, -2:] = data.loc[(data.y >= y_left) & (data.y < y_right)]. \
                                    groupby(level=['subj_id', 'trial_no']).mean(). \
                                    loc[:, ['vy','ay']].mean().values
        # if a cell has no data in it (NaN), derivatives in that cell remain zero
        # TODO: implement linear/constant extrapolation of available derivative data to empty cells
        mean_derivatives = np.nan_to_num(mean_derivatives)
        
        u_x = np.zeros(n_cells)
        u_y = np.zeros(n_cells)
        
        for i in range(n_cells):
            u_x[i] = np.trapz(-mean_derivatives[:i+1, 0] - \
                        gamma*mean_derivatives[:i+1, 1], dx = x_step)
            u_y[i] = np.trapz(-mean_derivatives[:i+1, 2] - \
                        gamma*mean_derivatives[:i+1, 3], dx = y_step)
        
        ds = np.tile(u_x, (n_cells, 1)) + np.tile(u_y, (n_cells, 1)).transpose()
        
        if return_u_xy:
            return ds, x_grid, y_grid, u_x, u_y
        else:
            return ds, x_grid, y_grid

    def generate_ds_cell_wise(self, data, n_cells=25, gamma=1.0):
        # generate decision space from the data containing x, y, v_x, v_y, a_x, a_y time series
        # using cell-wise averaging of time-derivatives, i.e., 
        # assuming general form of potential energy u = u(x,y)
        x_grid, x_step = np.linspace(self.x_lim[0], self.x_lim[1], n_cells+1, retstep=True)
        y_grid, y_step = np.linspace(self.y_lim[0], self.y_lim[1], n_cells+1, retstep=True)
        mean_derivatives = np.zeros((n_cells, n_cells, 4))
        for i in range(n_cells):
            [x_left, x_right] = [x_grid[i], x_grid[i+1]]
            for j in range(n_cells):
                [y_left, y_right] = [y_grid[j], y_grid[j+1]]
                # [j, i, :] is used instead of [i, j, :]  to map x coordinate onto 
                # 'horizontal' (column) index and y onto 'vertical' (row) index 
                # of the resulting array
                mean_derivatives[j, i, :] = data.loc[(data.x >= x_left) & (data.x < x_right) & \
                                                    (data.y >= y_left) & (data.y < y_right)]. \
                                                groupby(level=['subj_id', 'trial_no']).mean(). \
                                                loc[:, ['vx','ax','vy','ay']].mean().values       
        
        # if a cell has no data in it (NaN), derivatives in that cell remain zero
        # TODO: implement linear/constant extrapolation of available derivative data to empty cells
        mean_derivatives = np.nan_to_num(mean_derivatives)
        f_x = interp2d((x_grid[1:]+x_grid[:-1])/2, (y_grid[1:]+y_grid[:-1])/2,
                        -gamma*mean_derivatives[:, :, 0]-mean_derivatives[:, :, 1])
        f_y = interp2d((x_grid[1:]+x_grid[:-1])/2, (y_grid[1:]+y_grid[:-1])/2,
                        -gamma*mean_derivatives[:, :, 2]-mean_derivatives[:, :, 3])

        ds = np.zeros((n_cells, n_cells))        
        # for each x,y on the grid calculate the line integral of f_x and f_y 
        # along the straight line between (0,0) to (x,y)
        for i in range(n_cells):
            for j in range(n_cells):
                # parametrize 2-d functions f_x and f_y so that f_x_s and f_y_s are 1-d functions
                # s=0 corresponds to (0,0) and s=1 to (x_grid[i], y_grid[j])
                f_x_s = lambda s: f_x(s*x_grid[i], s*y_grid[j])
                f_y_s = lambda s: f_y(s*x_grid[i], s*y_grid[j])
                integrand = lambda s: np.dot([f_x_s(s)[0], f_y_s(s)[0]], [x_grid[i], y_grid[j]])
                ds[j, i], _ = integrate.quad(integrand, 0, 1)
                
        return ds, x_grid, y_grid
        
    def polyfit_ds(self, data, n_cells=25, gamma=1.0, retCoeff=False):
        # generate decision space surface and approximate it by polynomial 
        # (assuming x and y are independent)
        # TODO: do this for the cell-wise averaging
        # use np.polynomial.polynomial.polyval or np.linalg.lstsq        
        ds, x_grid, y_grid, u_x, u_y = self.generate_ds(data, n_cells, gamma, return_u_xy=True)
        
        p_x = np.polyfit((x_grid[:-1]+x_grid[1:])/2.0, u_x, 4)
        p_y = np.polyfit((y_grid[:-1]+y_grid[1:])/2.0, u_y, 3)
        u_x_fit = np.poly1d(p_x)
        u_y_fit = np.poly1d(p_y)
        
        ds_poly = np.tile(u_x_fit((x_grid[:-1]+x_grid[1:])/2.0), (n_cells, 1)) + \
            np.tile(u_y_fit((y_grid[:-1]+y_grid[1:])/2.0), (n_cells, 1)).transpose()        
        if retCoeff:
            return p_x, p_y, ds, ds_poly, x_grid, y_grid
        else:
            return ds, ds_poly, x_grid, y_grid
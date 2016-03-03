import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp2d

class DecisionSpaceGenerator:
    x_lim = [-275, 275]
    y_lim = [-50, 375]

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
                ds[j, i], _ = quad(integrand, 0, 1)
                
        return ds, x_grid, y_grid
        
    def polyfit_ds(self, data, n_cells=25, gamma=1.0):
        # generate decision space surface and approximate it by polynomial 
        # (assuming x and y are independent)
        # TODO: do this for the cell-wise averaging
        # use np.polynomial.polynomial.polyval or np.linalg.lstsq        
        ds, x_grid, y_grid, u_x, u_y = self.generate_ds(data, n_cells, gamma, return_u_xy=True)
        
        p_x = np.polyfit((x_grid[:-1]+x_grid[1:])/2.0, u_x, 4)
        p_y = np.polyfit((y_grid[:-1]+y_grid[1:])/2.0, u_y, 4)
        u_x_fit = np.poly1d(p_x)
        u_y_fit = np.poly1d(p_y)
        
        ds_poly = np.tile(u_x_fit((x_grid[:-1]+x_grid[1:])/2.0), (n_cells, 1)) + \
            np.tile(u_y_fit((y_grid[:-1]+y_grid[1:])/2.0), (n_cells, 1)).transpose()        
        
        return ds, ds_poly, x_grid, y_grid
import numpy as np
import scipy as sp
from scipy import integrate

class DLModelBase(object):
    '''
    This class defines the mathematical model for decision landscape in the form of 
    a system of two ordinary differential equations. It also encapsulates the definitions of
    fit error functions and its jacobians.
    
    Parameter alpha (2, 3, or 4) regulates the number of terms in the polynomial.
    '''       
    def model(self, z, t, coeffs):
        '''
        coeffs is an array of model parameters, indexed as
        0: tau, 1: c_11, 2: c_21, 3: c_12, 4: c_31, 5:c_22, 6:c_13
        '''
        x = z[0]
        y = z[1]
        
        x_dot = -self.dV_dx(x, y, coeffs)/coeffs[0]
        y_dot = -self.dV_dy(x, y, coeffs)/coeffs[0]
        return [x_dot, y_dot]
    
    def model_error(self, coeffs, trajectory):
        # defines fit error of the model for given set of coeffs with respect to one trajectory
        model_vx, model_vy = self.model([trajectory.x.values, trajectory.y.values], 0, coeffs)
        return ((model_vx - trajectory.vx.values)**2 + (model_vy - trajectory.vy.values)**2).mean()
    
    def model_error_multiple_traj(self, coeffs, trajectories):
        # defines fit error of the model with respect to multiple trajectories
        single_traj_error = lambda trajectory: self.model_error(coeffs, trajectory)
        return trajectories.groupby(level='trial_no').apply(single_traj_error).mean()
        
    def model_error_jac_multiple_traj(self, coeffs, trajectories):
        # jacobian of the fit error with respect to multiple trajectories
        single_traj_error_jac = lambda trajectory: self.model_error_jac(coeffs, trajectory)
        return trajectories.groupby(level = 'trial_no').apply(single_traj_error_jac).mean()
        
    def integrate_model_ode(self, coeffs, ic, time_grid):
        model = lambda z, t: self.model(z, t, coeffs)
        sol = integrate.odeint(model, ic, time_grid)
        return sol
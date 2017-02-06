import numpy as np
import scipy as sp
from scipy import integrate

class DSModel:
    def __init__(self, n_params=7):
        self.n_params = n_params
        
    def model(self, z, t, coeffs):
        # for n_params = 7, coeffs consist of 
        # 0: tau, 1: c_11, 2: c_21, 3: c_12, 4: c_31, 5:c_22, 6:c_13
        x = z[0]
        y = z[1]
        
        x_dot = -self.dV_dx(x, y, coeffs)/coeffs[0]
        y_dot = -self.dV_dy(x, y, coeffs)/coeffs[0]
        return [x_dot, y_dot]

    def dV_dx(self, x, y, coeffs):
        return (x - 1)*(x + 1)*x + coeffs[1]*y + coeffs[2]*x*y + (coeffs[3]/2)*y**2 + \
                coeffs[4]*x**2*y + coeffs[5]*x*y**2  + (coeffs[6]/3)*y**3
    
    def dV_dy(self, x, y, coeffs):
        return y*(y - 1) + coeffs[1]*x + (coeffs[2]/2)*x**2 + coeffs[3]*x*y + \
                (coeffs[4]/3)*x**3 + coeffs[5]*x**2*y + coeffs[6]*x*y**2
    
    def V(self, x, y, coeffs):
        V_x = x**4/4  - x**2/2
        V_y = y**3/3 - y**2/2
        V_xy = coeffs[1]*x*y + (coeffs[2]/2)*x**2*y + (coeffs[3]/2)*x*y**2 + \
                (coeffs[4]/3)*x**3*y + (coeffs[5]/2)*x**2*y**2  + (coeffs[6]/3)*x*y**3
        V = (V_x + V_y + V_xy)
        return V
        
    def solve_model(self, coeffs, ic, time_grid):
        model = lambda z, t: self.model(z, t, coeffs)
        sol = integrate.odeint(model, ic, time_grid)
        return sol
               
    def model_error(self, coeffs, data):
        model_vx, model_vy = self.model([data.x.values, data.y.values], 0, coeffs)
        return ((model_vx - data.vx.values)**2 + (model_vy - data.vy.values)**2).sum()
#        return (((model_vx - data.vx.values)/data.vx.abs().max())**2 + \
#                    ((model_vy - data.vy.values)/data.vy.abs().max())**2).sum()

    def model_error_jac(self, coeffs, data):
        model_vx, model_vy = self.model([data.x.values, data.y.values], 0, coeffs)        
        x, y, vx, vy = data.x.values, data.y.values, data.vx.values, data.vy.values
        
        left_operand = -2*np.vstack(((model_vx - vx)/coeffs[0], (model_vy - vy)/coeffs[0]))
                        
        de_dtau = (left_operand[0] * model_vx + left_operand[1] * model_vy).sum()
        de_dc11 = (left_operand[0] * y + left_operand[1] * x).sum()
        de_dc21 = (left_operand[0] * x*y + left_operand[1] * x**2/2).sum()
        de_dc12 = (left_operand[0] * y**2/2 + left_operand[1] * x*y).sum()
        de_dc31 = (left_operand[0] * x**2*y + left_operand[1] * x**3/3).sum()
        de_dc22 = (left_operand[0] * x*y**2 + left_operand[1] * x**2*y).sum()       
        de_dc13 = (left_operand[0] * y**3/3 + left_operand[1] * x*y**2).sum()
        
        return np.array([de_dtau, de_dc11, de_dc21, de_dc12, de_dc31, de_dc22, de_dc13])[:self.n_params]
    
    def model_error_multiple_traj(self, coeffs, trajectories):
        single_traj_error = lambda trajectory: self.model_error(coeffs, trajectory)
        return trajectories.groupby(level='trial_no').apply(single_traj_error).mean()
        
    def model_error_jac_multiple_traj(self, coeffs, trajectories):
        single_traj_error_jac = lambda trajectory: self.model_error_jac(coeffs, trajectory)
        return trajectories.groupby(level = 'trial_no').apply(single_traj_error_jac).mean()
    
    def get_param_names(self):
        return ['tau', 'c_11', 'c_21', 'c_12', 'c_31', 'c_22', 'c_13'][:self.n_params]
        
    def get_parameter_guess(self):
        return [0.05, 0, 0, 0, 0, 0, 0][:self.n_params]
    
    def get_parameter_bounds(self):
        k = 3.0
        return np.array([[0.005, 1.0], [-k, k], [-k, k],
                    [-k, k], [-k, k], [-k, k], [-k, k]])[:self.n_params]
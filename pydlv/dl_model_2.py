import numpy as np
from pydlv.dl_model_base import DLModelBase

class DLModel2(DLModelBase):
    '''
    This class defines the mathematical model for decision landscape in the form of 
    a system of two ordinary differential equations. It also encapsulates the definitions of
    fit error functions and its jacobians.
    
    coeffs is an array of model parameters, indexed as
    0: tau, 1: c_11
    '''

    def __init__(self):
        super(DLModelBase, self).__init__()
        self.n_params = 2
    
    def dV_dx(self, x, y, coeffs):  
        return (x - 1)*(x + 1)*x + coeffs[1]*y
    
    def dV_dy(self, x, y, coeffs):
        return (y - 1)*y + coeffs[1]*x

    def V(self, x, y, coeffs):
        V_x = x**4/4  - x**2/2
        V_y = y**3/3 - y**2/2
        V_xy = coeffs[1]*x*y
        V = (V_x + V_y + V_xy)
        return V

    def model_error_jac(self, coeffs, trajectory):
        # defines jacobian of the fit error function, which is needed for some fitting algorithms
        model_vx, model_vy = self.model([trajectory.x.values, trajectory.y.values], 0, coeffs)        
        x, y, vx, vy = trajectory.x.values, trajectory.y.values, \
                        trajectory.vx.values, trajectory.vy.values
        
        left_operand = -2*np.vstack(((model_vx - vx)/coeffs[0], (model_vy - vy)/coeffs[0]))
                        
        de_dtau = (left_operand[0] * model_vx + left_operand[1] * model_vy).mean()
        de_dc11 = (left_operand[0] * y + left_operand[1] * x).mean()
        
        return np.array([de_dtau, de_dc11])
   
    def get_param_names(self):
        return ['tau', 'c_11']

    def get_baseline_params(self):
        # baseline values of the model parameters
        return [0.05, 0]
    
    def get_parameter_bounds(self):
        '''
        Some fitting algorithms rely on bounded optimization methods (including recommended 
        L-BFGS-B). This function defines the bounds for each parameter.
        '''
        k = 3.0
        return np.array([[0.005, 1.0], [-k, k]])
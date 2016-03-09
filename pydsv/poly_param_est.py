import PyDSTool as dst
import numpy as np
import seaborn as sns
import maths

DSargs = dst.args(name='Decision space polynomial model')
DSargs.pars = { 'alpha_1': 1.0,
               'alpha_2': 1.0,
               'alpha_3': 1.0,
               'alpha_4': 1.0,
               'beta_1': 1.0,
               'beta_2': 1.0,
               'beta_3': 1.0 }
               
DSargs.varspecs = {'x': 'alpha_1*x + alpha_2*x**2 + alpha_3*x**3 + alpha_4*x**4',
                   'y': 'beta_1*y + beta_2*y**2 + beta_3*y**3'}

DSargs.ics = {'x': 0.01,
              'y': 0.01}
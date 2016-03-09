import pandas as pd
import numpy as np

class DerivativeCalculator:  
    def append_derivatives(self, data):
        return data.groupby(level=['subj_id', 'trial_no'], group_keys=False). \
                    apply(self.get_v_and_a)
    
    def get_v_and_a(self, trajectory):
        vx, ax = self.get_derivatives(trajectory.t.values, trajectory.x.values)
        vy, ay = self.get_derivatives(trajectory.t.values, trajectory.y.values)
                
        derivatives = pd.DataFrame(np.asarray([vx, vy, ax, ay]).T, 
                                   columns=['vx', 'vy', 'ax', 'ay'],
                                   index=trajectory.index)
        return pd.concat([trajectory, derivatives], axis=1)
    
    def get_derivatives(self, t, x):
        step = (t[1]-t[0])
        # HACK!
        # To be able to reasonably calculate derivatives at the end-points of the trajectories,
        # I append three extra points before and after the actual trajectory, so we get N+6
        # points instead of N       
        x = np.append(x[0]*np.ones(3),np.append(x, x[-1]*np.ones(3)))

        # smooth noise-robust differentiators: http://www.holoborodko.com/pavel/numerical-methods/ \
        # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
        v = (-x[:-6] - 4*x[1:-5] - 5*x[2:-4] + 5*x[4:-2] + 4*x[5:-1] + x[6:])/(32*step)
        a = (x[:-6] + 2*x[1:-5] - x[2:-4] - 4*x[3:-3] - x[4:-2] + 2*x[5:-1]+x[6:])\
                /(16*step*step)
        return v, a        
import pandas as pd
import numpy as np

#TODO: separate reading and pre-processing of the data
class DataReader:
    n_steps = 50
    lowHighTrials = [12, 15, 25]
    
    def read_data(self):
        dynamicsPath = '../2013_data/pod_traj_inc3.txt'
        choicesPath = '../2013_data/pod_choice_inc3.txt'
        
        dynamics = pd.read_csv(dynamicsPath, sep=' ', 
            names=['subj_id', 'trial_no', 't', 'x', 'y'],
            header=None)
        dynamics.set_index(['subj_id', 'trial_no'], inplace=True, drop=False)
        
        choices = pd.read_csv(choicesPath, sep=' ', 
            names=['subj_id', 'trial_no', 'rewards_sum', 'symbol', 'outcome', 'resp_time'], 
            header=None)
        choices.set_index(['subj_id', 'trial_no'], inplace=True)

        data = pd.merge(dynamics, choices, left_index=True, right_index=True)
        data = data.groupby(level='subj_id').apply(self.add_exp_type)
        return data
        
    def add_exp_type(self, data):
        # exp_type contains the high-value choice (7, 10, or 20) 
        # for each trajectory (including 'low-low' ones)
        data['exp_type'] = data.rewards_sum.max()/2
        return data        
    
    def preprocess_data(self, data, exp_type=None, rewards_sum=None, trial_nos=None):
        # Trim the dataset to specific condition (7/5, 10/5, etc.)
        # If needed, trial number can be filtered at this stage as well (e.g., last third of trials)
        if exp_type !=None:
            data = data.loc[data['exp_type']==exp_type]
        if rewards_sum !=None:
            data = data.loc[data['rewards_sum']==rewards_sum]
        if trial_nos != None:
            data = data.loc[(data['trial_no']>=trial_nos[0]) & (data['trial_no']<=trial_nos[1])]
        
        # For convenience, store subject ids separately
        subj_ids = data.index.get_level_values('subj_id').unique()

        # Move starting point to 0 and invert y axis
        startingPoints = data.groupby(level=['subj_id', 'trial_no']).first()
        data.x -= startingPoints.mean(axis = 0).x
        data.y = startingPoints.mean(axis = 0).y - data.y

        # Remove reaction time: keep only last occurrence of the same x,y for each trial
        data = data.groupby(level=['subj_id', 'trial_no'], group_keys=False). \
            apply(lambda d: d.drop_duplicates(subset=['x', 'y'], keep='last'))
        # shift time to the timeframe beginning at 0 for each trajectory
        data.t = data.groupby(level=['subj_id', 'trial_no'])['t']. \
            apply(lambda t: (t-t.min()))
        
        # Make sure all 'high' options are mapped onto the right-hand side of the screen 
        data = data.groupby(level=['subj_id', 'trial_no']).apply(self.reverse_x)
        # Resample trajectories so that each trajectory has n_steps points
        data = data.groupby(level=['subj_id', 'trial_no']).apply(self.resample_trajectory)
        # ToDo: this is hack, think about how to properly get rid of extra index in interpolate_xy
        data.index = data.index.droplevel(2)
            
        return subj_ids, data
   
    def reverse_x(self, trajectory):
        # We need to reverse those 'high-low' trajectories which have 'high' option on the left
        # As the data doesn't explicitly specify the location of the presented options, we have to  
        # search for those trials which either 1) have 'high' outcome and the final point is on the 
        # left, or 2) have 'low' outcome and the final point on the right
    
        # Only reverse if trajectory is 'high-low'
        if (trajectory.iloc[0]['rewards_sum'] == trajectory.iloc[0]['exp_type'] + 5):
            is_final_point_positive = trajectory.iloc[-1]['x']>0
            is_outcome_high = trajectory.iloc[-1]['outcome']>trajectory.iloc[-1]['rewards_sum']/2
            if (is_final_point_positive != is_outcome_high):
                trajectory.loc[:,'x'] = -trajectory.loc[:,'x']
        return trajectory
        
    def resample_trajectory(self, trajectory):
        # Make the sampling time intervals regular
        t_regular = np.linspace(trajectory.t.min(), trajectory.t.max(), self.n_steps+1)
        x_interp = np.interp(t_regular, trajectory.t.values, trajectory.x.values)
        y_interp = np.interp(t_regular, trajectory.t.values, trajectory.y.values)
        
        traj_interp = pd.DataFrame([t_regular, x_interp, y_interp]).transpose()
        traj_interp.columns = ['t', 'x', 'y']
        return traj_interp

    def append_derivatives(self, data):
        return data.groupby(level=['subj_id', 'trial_no'], group_keys=False). \
                    apply(self.get_v_and_a)
    
    def get_v_and_a(self, trajectory):
        t = trajectory.t.values
        x = trajectory.x.values
        y = trajectory.y.values
        
        vx, ax = self.get_derivatives(t, x)
        vy, ay = self.get_derivatives(t, y)
                
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
        
    def zero_cross_count(self, x):
        return (np.diff(np.sign(x)) != 0).sum()    
        
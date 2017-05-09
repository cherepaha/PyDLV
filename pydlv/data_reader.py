import pandas as pd
import numpy as np

#TODO: separate reading and pre-processing of the data
class DataReader:
    
    '''
    The input trajectories are assumed to be time-normalised, augmented by mouse x- and 
    y-velocities, and presented in a 'long' dataframe format. It should contain the following 
    columns (in any order): subj_id, trial_no, t, x, y, vx, vy, chng_mind, xflip_count.
    t, x, y, vx and vy columns are: time, mouse x- and y- coordinates, mouse x- and y- velocities
    chng_mind is Boolean column indicating whether the trajectory has change-of-mind or not
    xflip_count indicates the number of changes of sign of mouse x-velocity
    '''
    
    def read_data(self, path):
        dynamicsPath = path + '/pod_traj_inc3.txt'
        choicesPath = path + '/pod_choice_inc3.txt'
        
        dynamics = pd.read_csv(dynamicsPath, sep=' ', 
            names=['subj_id', 'trial_no', 't', 'x', 'y'],
            header=None)
        dynamics.set_index(['subj_id', 'trial_no'], inplace=True, drop=True)
        
        choices = pd.read_csv(choicesPath, sep=' ', 
            names=['subj_id', 'trial_no', 'rewards_sum', 'symbol', 'outcome', 'resp_time'], 
            header=None)
        choices.set_index(['subj_id', 'trial_no'], inplace=True, drop=True)
        
        data = pd.merge(dynamics, choices, left_index=True, right_index=True)
        data = data.groupby(level='subj_id').apply(self.add_exp_type)
        # high_chosen is False for a given trial if 'low' option was chosen 
        # and True if 'high' was chosen
        data.loc[:,'high_chosen'] = (data.outcome != 5)
        
        return data
    
    def add_exp_type(self, data):
        # exp_type contains the high-value choice (7, 10, or 20) 
        # for each trajectory (including 'low-low' ones)
        data['exp_type'] = data.rewards_sum.max()/2
        return data    
        
    def get_processed_data(self, path):
        data = pd.read_csv(path, sep=',', header=0)
        data.set_index(['subj_id', 'trial_no'], inplace=True, drop=False)
        data = data[(data.xflip_count < 4) & (data.chng_mind == False)]  

        # Partition the trials into three blocks, so that trials 1 to 12 belong to Block 1, etc.
        data.loc[:, 'block_no'] = pd.cut(data.trial_no, bins=np.linspace(0, 36, 4), 
                                    labels=[1, 2, 3], right=True)
        return data
            
    def preprocess_data(self, data, exp_type=None, rewards_sum=None):
        # exclude subjects with problematic data
        data = data.drop([2302, 3217], level = 'subj_id') 
                
        # Trim the dataset to specific condition (7/5, 10/5, etc.)
        # If needed, trial number can be filtered at this stage as well (e.g., last third of trials)
        if exp_type !=None:
            data = data.loc[data['exp_type'].isin(exp_type)]
        if rewards_sum !=None:
            data = data.loc[data['rewards_sum'].isin(rewards_sum)]
                            
        # Move starting point to 0 and invert y axis, then rescale trajectories to (-1,1) x (0,1)
        startingPoints = data.groupby(level=['subj_id', 'trial_no']).first()
        data.x = data.x - startingPoints.mean(axis = 0).x
        data.y = startingPoints.mean(axis = 0).y - data.y
        data = self.rescale_trajectories(data)
        
        # Remove reaction time: keep only last occurrence of the same x,y for each trial        
        data = data.groupby(level=['subj_id', 'trial_no'], group_keys=False). \
            apply(lambda df: df.drop_duplicates(subset=['x', 'y'], keep='last'))
        
        # Then, we need to delete last point in every trajectory
        data = data.groupby(level=['subj_id', 'trial_no'], group_keys=False). \
            apply(lambda df: df.ix[:-1])
        
        # shift time to the timeframe beginning at 0 for each trajectory
        data.loc[:, 't'] = data.t.groupby(level=['subj_id', 'trial_no']).apply(lambda t: (t-t.min()))

        data['resp_time'] = data['resp_time']/1000.0
        data['motion_time'] = data.t.groupby(level=['subj_id', 'trial_no']).max()
        
        # all 'high' options are re-mapped onto the right-hand side of the screen 
        data = data.groupby(level=['subj_id', 'trial_no']).apply(self.reverse_x)
        # Resample trajectories so that each trajectory has n_steps points
        data = data.groupby(level=['subj_id', 'trial_no']).apply(self.resample_trajectory)
        # ToDo: this is hack, think about how to properly get rid of extra index in resample_trajectory
        data.index = data.index.droplevel(2)
        
        data = data.groupby(level=['subj_id', 'trial_no']).apply(self.shift_starting_point)
        data = data.groupby(level=['subj_id', 'trial_no']).apply(self.get_maxd)
        data = data.groupby(level=['subj_id', 'trial_no']).apply(self.get_chng_mind)

        return data
        
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
        
    def resample_trajectory(self, trajectory, n_steps = 50):
        # Make the sampling time intervals regular
        t_regular = np.linspace(trajectory.t.min(), trajectory.t.max(), n_steps+1)
        x_interp = np.interp(t_regular, trajectory.t.values, trajectory.x.values)
        y_interp = np.interp(t_regular, trajectory.t.values, trajectory.y.values)
        
        traj_interp = pd.DataFrame([t_regular, x_interp, y_interp]).transpose()
        traj_interp.columns = ['t', 'x', 'y']
        for column in trajectory.columns[3:]:
            traj_interp[column] = trajectory[column].ix[0]
        return traj_interp
        
    def get_chng_mind(self, trajectory, threshold = 0.25):
        is_final_point_positive = trajectory.iloc[-1]['x']>0
        trajectory['chng_mind'] = False
        if is_final_point_positive:
            trajectory['midline_d'] = abs(trajectory.x.min())
            if trajectory.x.min() < -threshold:
                trajectory['chng_mind'] = True        
        else:
            trajectory['midline_d'] = abs(trajectory.x.max())
            if trajectory.x.max() > threshold:
                trajectory['chng_mind'] = True
        return trajectory

    def get_maxd(self, trajectory):
        alpha = np.arctan((trajectory.y.iloc[-1]-trajectory.y.iloc[0])/ \
                            (trajectory.x.iloc[-1]-trajectory.x.iloc[0]))
        d = (trajectory.x.values-trajectory.x.values[0])*np.sin(-alpha) + \
            (trajectory.y.values-trajectory.y.values[0])*np.cos(-alpha)
        trajectory['max_d'] = max(d.min(), d.max(), key=abs)
        return trajectory

    def shift_starting_point(self, trajectory):
        if trajectory.x.iloc[0]*trajectory.x.iloc[-1] < 0:
            trajectory.x -= trajectory.x.iloc[0]
        return trajectory
    
    def rescale_trajectories(self, trajectories):
        #change frame of reference so that stimuli are located at (-1, 1) and (1, 1)
        stim_x = 195
        stim_y = 280
        trajectories.loc[:,['t']] /= 1000
        trajectories.loc[:, ['x', 'y']] /= [stim_x, stim_y]
        return trajectories
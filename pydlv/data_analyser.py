import numpy as np

class DataAnalyser:
    def get_subjects_stats(self, data):
        stats = data.groupby([data.index.get_level_values(0), 'high_chosen']).high_chosen.count().\
            unstack(level=-1, fill_value=0)/51
        return stats
        
    def get_block_stats(self, data):
        stats = data.groupby([data.index.get_level_values(0), 'block_no', 'high_chosen']).high_chosen.count().\
            unstack(level=-1, fill_value=0)/51
        return stats
        
    def get_mean_trajectories(self, trajectories, by=['high_chosen']):
        cols = trajectories.index.names + by
        mouse_coordinates = trajectories.groupby(by = cols). \
                        apply(lambda x: x.loc[:, ['x', 'y']].reset_index(drop=True).T)
        cols += ['x/y']
        mouse_coordinates.index.names = cols
        subj_mean_traj = mouse_coordinates.groupby(level = by + ['x/y']).apply(np.mean).stack()
        subj_mean_traj.index.names = by + ['x/y', 't']
        subj_mean_traj = subj_mean_traj.unstack('x/y')
        
        return subj_mean_traj
    

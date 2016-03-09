import data_reader, trajectory_plotter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

exp_type = 20

#High/High
#rewards_sum = exp_type*2

#High/Low
rewards_sum = exp_type + 5

#Low/Low
#rewards_sum = 10

trial_nos = [24, 36]

reader = data_reader.DataReader()

data = reader.read_data()
subj_ids, proc_data = reader.preprocess_data(data, exp_type, rewards_sum, trial_nos)

subj_id = subj_ids[0]
trial_no = 27

trajectory = proc_data.loc[(subj_id, trial_no)]
file_name = 'trajectory_subj_id_'+str(subj_id)+'_trial_'+str(trial_no)+'.csv'
#trajectory.to_csv(file_name, sep='\t')

tp = trajectory_plotter.TrajectoryPlotter()
#tp.plot_subject_trajectories(proc_data, subj_id)
tp.plot_trajectory(proc_data, subj_id, trial_no)

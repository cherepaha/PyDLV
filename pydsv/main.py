import data_reader
import decision_space_generator
import ds_plotter
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
dsg = decision_space_generator.DecisionSpaceGenerator()

data = reader.read_data()
subj_ids, proc_data = reader.preprocess_data(data, exp_type, rewards_sum, trial_nos)
proc_data = reader.append_derivatives(proc_data)
#ds, x_grid, y_grid = dsg.generate_ds_cell_wise(proc_data)
ds, x_grid, y_grid = dsg.generate_ds(proc_data, gamma=1.0)

plotter = ds_plotter.DSPlotter()
plotter.plot_surface(x_grid, y_grid, ds, cmap=cm.PuRd_r)

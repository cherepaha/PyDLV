import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from pydlv import dl_model_3, dl_generator, dl_plotter, data_reader, trajectory_plotter

'''
This script demonstrates how, using coefficients from the csv file generated previously,
plot a 3d surface of decision landscape fitted to individual trajectories.
'''
    
def plot_surfaces(dlg, params, subj_id, trials, colors):
    dlp = dl_plotter.DLPlotter(elev=10, azim=61)
    for i, trial_no in enumerate(trials):
        x, y, dl = dlg.get_model_dl(params.loc[subj_id, trial_no][2:2+dlg.model.n_params])
        dlp.plot_surface(x, y, dl, color=colors[i], alpha=0.8) 
    dlp.add_legend(colors, trials)
    plt.savefig('figures/trials_%i_dlv.pdf' % (subj_id))

def plot_trajectories(data, subj_id, trials, colors):
    tp = trajectory_plotter.TrajectoryPlotter()
    for i, trial_no in enumerate(trials):
        trial_data = data[(data.subj_id==subj_id) & (data.trial_no==trial_no)]
        tp.plot_trajectory(trial_data, color=colors[i], label=trial_no)
        print(trial_data .iloc[0][['high_chosen', 'motion_time', 'max_d']])
    tp.add_legend()
    plt.savefig('figures/trials_%i_traj.pdf' % (subj_id))

def compare_dlv(subj_id, trials):
    fit_params = pd.read_csv('csv/fit_params_by_trial_method_9.csv', 
                             index_col=['subj_id', 'trial_no'], header=0)
    cmap = cm.viridis
    colors = [cmap(0.7), cmap(0.35), cmap(0.1)]
          
    model = dl_model_3.DLModel3()  
    dlg = dl_generator.DLGenerator(model) 
    plot_surfaces(dlg, fit_params, subj_id, trials, colors)
    
    dr = data_reader.DataReader()
    data = dr.get_processed_data(path='csv/processed_data_high_low.csv')
    plot_trajectories(data, subj_id, trials, colors)
    
subj_id = 90
trials = [28, 2, 15]
compare_dlv(subj_id, trials)
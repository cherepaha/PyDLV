import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from pydlv import dl_model_3, data_reader, data_analyser, dl_generator, dl_plotter, trajectory_plotter

'''
This script demonstrates how, using coefficients from the csv file generated previously,
plot 3d decision landscapes fitted to individual participants' data.
'''

def plot_surfaces(dlg, fit_params, subjects, colors, labels):
    dlp = dl_plotter.DLPlotter(elev=33, azim=107)
    for i, subj_id in enumerate(subjects):
        x, y, dl = dlg.get_model_dl(fit_params.loc[subj_id][2:2+dlg.model.n_params])
        dlp.plot_surface(x, y, dl, color=colors[i], alpha=0.8)
    dlp.add_legend(colors, labels)
    plt.savefig('figures/subjects_%i_%i_dlv.pdf' % (subjects[0], subjects[1]))

def plot_trajectories(data, subjects, colors, labels):
    tp = trajectory_plotter.TrajectoryPlotter()
    for i, subj_id in enumerate(subjects):
        tp.plot_mean_trajectories(data.loc[subj_id], colors[i], labels[i])
        subj_info = data.loc[subj_id, ['high_chosen', 'motion_time', 'max_d']].\
                        groupby(level='trial_no').first().groupby('high_chosen').mean()
        print('\n %s\n' % (labels[i]))
        print(subj_info)      
    tp.add_legend_mean_traj(colors, labels)
    plt.savefig('figures/subjects_%i_%i_traj.pdf' % (subjects[0], subjects[1]))

def compare_dlv(subjects):
    fit_params = pd.read_csv('csv/fit_params_by_subject_method_9.csv', 
                             index_col='subj_id', header=0)
    labels = ['Participant %i' % (subj_id) for subj_id in subjects]
    
    cmap = cm.viridis
    colors = [cmap(0.1), cmap(0.7)]
    
    model = dl_model_3.DLModel3()   
    dlg = dl_generator.DLGenerator(model)
    plot_surfaces(dlg, fit_params, subjects, colors, labels)
    
    dr = data_reader.DataReader()
    data = dr.get_processed_data(path='csv/processed_data_high_low.csv')
    plot_trajectories(data, subjects, colors, labels)
    
    da = data_analyser.DataAnalyser()
    stats = da.get_subjects_stats(data)
    print(stats.loc[subjects])

compare_dlv(subjects=[9276, 9424])
compare_dlv(subjects=[1395, 1962])
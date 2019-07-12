import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from pydlv import dl_model_3, data_reader, data_analyser, dl_generator, dl_plotter, trajectory_plotter

'''
This script demonstrates how, using coefficients from the csv file generated previously,
plot a 3d surface of decision landscapes fitted to blocks of trajectories.
'''

def plot_surfaces(dlg, fit_params, subj_id, blocks, colors, labels):
    dlp = dl_plotter.DLPlotter(elev=10, azim=69)
    for i, block_no in enumerate(blocks):
        x, y, dl = dlg.get_model_dl(fit_params.loc[subj_id, block_no][2:2+dlg.model.n_params])
        dlp.plot_surface(x, y, dl, color=colors[i], alpha=0.8)
    dlp.add_legend(colors, labels)
    plt.savefig('figures/blocks_%i_dlv.pdf' % (subj_id))

def plot_trajectories(data, subj_id, blocks, colors, labels):
    tp = trajectory_plotter.TrajectoryPlotter()
    
    for i, block_no in enumerate(blocks):
        block_trajectories = data[(data.subj_id==subj_id) & (data.block_no==block_no)]
        tp.plot_mean_trajectories(block_trajectories, colors[i], labels[i])
        block_info = block_trajectories.groupby('trial_no').first().groupby('high_chosen') \
                .mean()[['motion_time', 'max_d']]
        print('\n %s\n' % (labels[i]))
        print(block_info)
    tp.add_legend_mean_traj(colors, labels)
    plt.savefig('figures/blocks_%i_traj.pdf' % (subj_id))

def compare_dlv(subj_id, blocks):
    fit_params = pd.read_csv('csv/fit_params_by_block_method_9.csv', 
                             index_col=['subj_id', 'block_no'], header=0)    
    labels = ['Block %i' % (block) for block in blocks]
    
    cmap = cm.viridis
    colors = [cmap(0.7), cmap(0.35), cmap(0.1)]
    
    model = dl_model_3.DLModel3()   
    dlg = dl_generator.DLGenerator(model)
    plot_surfaces(dlg, fit_params, subj_id, blocks, colors, labels)
    
    dr = data_reader.DataReader()
    data = dr.get_processed_data(path='csv/processed_data_high_low.csv')
    plot_trajectories(data, subj_id, blocks, colors, labels)
    
    da = data_analyser.DataAnalyser()
    stats = da.get_block_stats(data)
    print('\n %s\n' % ('Block stats'))
    print(stats.loc[subj_id])
    
#subj_id = 233
subj_id = 1334
blocks = [1, 2, 3]
compare_dlv(subj_id, blocks)
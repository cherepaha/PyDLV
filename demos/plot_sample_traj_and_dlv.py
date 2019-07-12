from pydlv import data_reader, dl_model_3, dl_generator, dl_plotter
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpl_patches
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from scipy import interpolate
import numpy as np

def get_choice_patches():
    screen_patch = mpl_patches.Rectangle((-1.1, -0.1), 2.2, 1.3, fill=False, lw=3, 
                                         edgecolor='black')
    left_patch = mpl_patches.Rectangle((-1.05, 0.85), 0.3, 0.3, fill=True, lw=1, 
                                   facecolor='white', edgecolor='black', alpha=0.2)
    right_patch = mpl_patches.Rectangle((0.75, 0.85), 0.3, 0.3, fill=True, lw=1, 
                                   facecolor='white', edgecolor='black', alpha=0.2)
    left_text = mpl_patches.PathPatch(TextPath((-1.0, 0.9), 'A', size=0.3, 
                                               backgroundcolor='white'))
    right_text = mpl_patches.PathPatch(TextPath((0.8, 0.9), 'B', size=0.3))
    
    return screen_patch, left_patch, right_patch, left_text, right_text

def plot_sample_trajectory(data, subj_id, trial_no):
    x_lim = [-1.2, 1.2]
    y_lim = [-0.2, 1.2]
    
    tickLabelFontSize = 20
    axisLabelFontSize = 24

    sns.set_style('white')    
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'x coordinate', fontsize=axisLabelFontSize)
    ax.set_ylabel(r'y coordinate', fontsize=axisLabelFontSize)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=tickLabelFontSize)
    
    traj_color = cm.viridis(0.1)

    trajectory = data[(data.subj_id==subj_id) & (data.trial_no==trial_no)]
    ax.plot(trajectory.x, trajectory.y, color=traj_color, ls='none', marker='o', ms=15,
            markerfacecolor='none', markeredgewidth=2, markeredgecolor=traj_color, 
            label='Mouse trajectory')

    # draw screen above the surface and choice options on it
    patches = get_choice_patches()
    for patch in patches:
        ax.add_patch(patch)
        
    ax.set_axis_off()
    plt.savefig('figures/sample_traj.pdf')

def plot_baseline_landscape_overlay(dlg, data, subj_id, trial_no):  
    dlp = dl_plotter.DLPlotter(elev=55, azim=-65)
        
    x_grid, y_grid, dl = dlg.get_model_dl(dlg.model.get_baseline_params()*4)
    x=(x_grid[1:]+x_grid[:-1])/2
    y=(y_grid[1:]+y_grid[:-1])/2
    f = interpolate.interp2d(x,y,dl,kind='cubic')
    
    ax = dlp.plot_surface(x_grid, y_grid, dl, cmap=cm.viridis, alpha=0.9)
    
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.w_zaxis.set_ticklabels([])
    
    sns.set_style('white')
    
    cmap = cm.viridis
    traj_color = cmap(0.1)    
    trajectory = data[(data.subj_id==subj_id) & (data.trial_no==trial_no)]
    z = f(trajectory.x.values, trajectory.y.values)
    if trajectory.x.values[-1]>0:
        z= np.diag(z)
    else:
        z=np.diag(np.fliplr(z))
    
    # plot trajectory on a surface
    ax.plot(trajectory.x.values, trajectory.y.values, z, color='black', alpha=0.5)
    
    # plot marble 
    ax.plot([0.], [0.], [0.], marker='o', markersize=15, color = 'black', alpha=0.7)
    ax.plot([trajectory.x.values[-1]], [trajectory.y.values[-1]], [z[-1]], 
            marker='o', markersize=15, color='black', alpha=0.7)
#    
    # draw screen above the surface and choice options on it
    patches = get_choice_patches()
    for patch in patches:
        ax.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch, z=0, zdir='z')
        
    # plot trajectory on a screen
    ax.plot(trajectory.x, trajectory.y, zs=0, zdir='z', color=traj_color, ls='none', 
            alpha=1.0, marker='o', ms=15, markerfacecolor='none', markeredgewidth=2, 
            markeredgecolor=traj_color, label='Mouse trajectory')
    plt.savefig('figures/baseline_dlv.pdf')

model = dl_model_3.DLModel3()
dlg = dl_generator.DLGenerator(model)    

dr = data_reader.DataReader()
data = dr.get_processed_data(path='csv/processed_data_high_low.csv')

plot_sample_trajectory(data=data, subj_id=8792, trial_no=13)
plot_baseline_landscape_overlay(dlg=dlg, data=data, subj_id=8792, trial_no=13)
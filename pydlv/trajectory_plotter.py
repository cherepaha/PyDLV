import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import data_analyser

class TrajectoryPlotter:
    x_lim = [-1.35, 1.35]
    y_lim = [-0.2, 1.2]
    
    lw = 2.0
    mean_lw = 5.0
    
    legendFontSize = 20
    tickLabelFontSize = 20
    axisLabelFontSize = 24
    
    def __init__(self, ax=None):
        sns.set_style('white')
        if ax is None:
            self.ax = plt.figure(tight_layout=True).add_subplot(111)
        else:
            self.ax = ax
        self.set_axis_params()
     
    def set_axis_params(self):
        self.ax.set_xlabel(r'x coordinate', fontsize=self.axisLabelFontSize)
        self.ax.set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize)
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tickLabelFontSize)
    
    def plot_trajectory(self, trajectory, color='k', label='Trajectory', marker='o', lw=2.0, alpha=1.0):
        self.ax.plot(trajectory.x.values, trajectory.y.values, color=color, label=label, 
                     marker=marker, lw=lw, alpha=alpha)
        
    def plot_mean_trajectories(self, trajectories, color, label):
        # plots all provided trajectories and two mean trajectories, for 'high' and 'low' choices
        for idx, trajectory in trajectories.groupby(level='trial_no'):
            self.plot_trajectory(trajectory, color, label, marker='None', lw=self.lw, alpha=0.5)
        
        da = data_analyser.DataAnalyser()
        mean_trajectories = da.get_mean_trajectories(trajectories)
        for condition in mean_trajectories.index.get_level_values(0).unique():
            x = mean_trajectories.loc[condition].x.values
            y = mean_trajectories.loc[condition].y.values
            self.ax.plot(x, y, marker='o', color=color, label=label, lw=self.mean_lw, ms=10.0)
        
    def add_legend(self):
        self.ax.legend(fontsize=self.legendFontSize)
    
    def add_legend_mean_traj(self, colors, labels):
        lines = [mpl.lines.Line2D([], [], color=color, lw=self.mean_lw) for color in colors]
        self.ax.legend(lines, labels, fontsize=self.legendFontSize)
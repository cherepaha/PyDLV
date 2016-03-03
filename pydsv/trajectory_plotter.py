import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class TrajectoryPlotter:
    x_lim = [-275, 275]
    y_lim = [-50, 400]
    
    n_cells = 30
    legendFontSize = 15
    tickLabelFontSize = 20
    axisLabelFontSize = 24
    lw=2.0
    
    def __init__(self):
        plt.ioff()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.set_axis_params()     
#        sns.set_palette('Set3', 10)
     
    def set_axis_params(self):
        self.ax.set_xlabel(r'x coordinate', fontsize=self.axisLabelFontSize)
        self.ax.set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize)
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tickLabelFontSize)
    
    def plot_trajectory(self, data, subj_id, trial_no):
        x = data.loc[(subj_id, trial_no)].x.values
        y = data.loc[(subj_id, trial_no)].y.values
        self.ax.plot(x, y, marker='o', label=trial_no)
        plt.tight_layout()
        
    def plot_subject_trajectories(self, data, subj_id):
        for trial_no, trajectory in data.loc[subj_id].groupby(level='trial_no'):
            self.ax.plot(trajectory.x.values, trajectory.y.values, marker='o')
        self.ax.set_title('Participant '+ str(subj_id), fontsize=self.axisLabelFontSize)
#        self.add_grid()
        plt.tight_layout()
        
    def plot_all_subject_trajectories(self, data, subj_ids):
        self.set_axis_params()        
        for subj_id in subj_ids:
            for trial_no, trajectory in data.loc[subj_id].groupby(level='trial_no'):
                self.ax.plot(trajectory.x.values, trajectory.y.values, marker='o')
            self.ax.set_title('Participant '+ str(subj_id), fontsize=self.axisLabelFontSize)
    #        self.add_grid()
            plt.tight_layout()
            plt.savefig(str(subj_id)+'.png')
            plt.cla()
#        plt.ion()
        
    def add_grid(self):
        x_ticks = np.linspace(self.x_lim[0], self.x_lim[1], self.n_cells)
        y_ticks = np.linspace(self.y_lim[0], self.y_lim[1], self.n_cells)
        
        self.ax.set_xticks(x_ticks, minor=True)
        self.ax.set_yticks(y_ticks, minor=True)

        self.ax.grid(b=False, which='major')
        self.ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    
    def plot_surface(self, x, y, z):
        self.ax = self.fig.gca(projection='3d')
        x, y = np.meshgrid(x, y)
        
        

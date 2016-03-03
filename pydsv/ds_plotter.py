import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

class DSPlotter:
    x_lim = [-275, 275]
    y_lim = [-50, 375]
    z_lim = [-400, 150]
    n_cells = 30
    
    legendFontSize = 15
    tickLabelFontSize = 14
    axisLabelFontSize = 18
    lw=2.0
    
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.set_axis_params()     
#        sns.set_palette('Set3', 10)
     
    def set_axis_params(self):
        self.ax.xaxis.set_major_locator(MaxNLocator(5))
        self.ax.yaxis.set_major_locator(MaxNLocator(5))
        self.ax.zaxis.set_major_locator(MaxNLocator(1))
        self.ax.set_xlabel(r'x coordinate', fontsize=self.axisLabelFontSize, labelpad=10)
        self.ax.set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize, labelpad=10)
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_zlim(self.z_lim)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tickLabelFontSize)
    
    def plot_surface(self, x_grid, y_grid, z, cmap=cm.jet):
        x, y = np.meshgrid((x_grid[1:]+x_grid[:-1])/2, (y_grid[1:]+y_grid[:-1])/2)
        z = np.nan_to_num(z)
        norm = mpl.colors.Normalize(vmin = np.min(z), vmax = np.max(z), clip=False)
        self.ax.plot_surface(x, y, z, cmap=cmap, norm=norm, alpha=0.5,
                             rstride=1, cstride=1, linewidth=0, antialiased=False)
#        plt.tight_layout()
        

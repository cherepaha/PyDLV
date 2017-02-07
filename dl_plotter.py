import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy import interpolate

class DLPlotter:
    '''
    This class is responsible for plotting decision landscapes. Matplotlib is used as a background.
    '''
    figsize = (14, 8) # in hundreds of pixels
    
    legendFontSize = 24
    tickLabelFontSize = 18
    axisLabelFontSize = 24
    lw=2.0
    
    def __init__(self, elev=27, azim=130):
        plt.ion()
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.gca(projection='3d')
        self.set_axis_params(elev, azim)
        sns.set_style('white')
     
    def set_axis_params(self, elev=27, azim=130):
        self.ax.xaxis.set_major_locator(MaxNLocator(5))
        self.ax.yaxis.set_major_locator(MaxNLocator(5))
        self.ax.zaxis.set_major_locator(MaxNLocator(1))
        self.ax.set_xlabel(r'x coordinate', fontsize=self.axisLabelFontSize, labelpad=10)
        self.ax.set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize, labelpad=10)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tickLabelFontSize)
        self.ax.view_init(elev, azim) 
        
    def plot_surface(self, x_grid, y_grid, z, cmap=None, color=None, scale_z=True, view='top left', 
                     alpha=1.0):
        x, y = np.meshgrid((x_grid[1:]+x_grid[:-1])/2, (y_grid[1:]+y_grid[:-1])/2)
        z = np.nan_to_num(z)
        if scale_z:
            self.ax.set_zlim([np.min(z), 0])
        norm = mpl.colors.Normalize(vmin=np.min(z), vmax=0, clip=False)
        self.ax.plot([0.], [0.], [0.], marker='o', markersize=15, color='black')
        if not color is None:
            self.ax.plot_surface(x, y, z, color=color, norm=norm, alpha=alpha,
                             rstride=1, cstride=1, linewidth=0, antialiased=True)
        elif not cmap is None:
            self.ax.plot_surface(x, y, z, cmap=cmap, norm=norm, alpha=alpha,
                             rstride=1, cstride=1, linewidth=1, antialiased=True)
        else: 
            self.ax.plot_surface(x, y, z, cmap=cm.jet, norm=norm, alpha=alpha,
                             rstride=1, cstride=1, linewidth=0, antialiased=True)           
        if view == 'top right':
            self.ax.view_init(elev=27, azim=40)
        
        plt.tight_layout()
        return self.ax
    
    def add_legend(self, colors, labels):
        patches = []
        for color in colors:
            patch = mpl.patches.Patch(color=color, linewidth=0)
            patches.append(patch)
        self.ax.legend(patches, labels, fontsize=self.legendFontSize)      
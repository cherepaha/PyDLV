import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

class DLPlotter:
    '''
    This class is responsible for plotting decision landscapes. Matplotlib is used as a background.
    '''
    figsize = (10.5, 6) # in inches, at 100 dpi
#    figsize = (14, 8) # in inches, at 100 dpi
    
    legendFontSize = 24
    tickLabelFontSize = 18
    axisLabelFontSize = 24
    lw=2.0
    
    def __init__(self, elev=27, azim=130, ax=None):  
        if ax is None:
            fig = plt.figure(figsize=self.figsize, tight_layout=True)
            self.ax = fig.gca(projection='3d')
        else:
            self.ax = ax
        self.set_axis_params(elev, azim)
     
    def set_axis_params(self, elev=27, azim=130):
        self.ax.xaxis.set_major_locator(MaxNLocator(5))
        self.ax.yaxis.set_major_locator(MaxNLocator(5))
        self.ax.zaxis.set_major_locator(MaxNLocator(1))
        self.ax.set_xlabel(r'x coordinate', fontsize=self.axisLabelFontSize, labelpad=20)
        self.ax.set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize, labelpad=20)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tickLabelFontSize)
        self.ax.view_init(elev, azim) 
        
    def plot_surface(self, x_grid, y_grid, z, cmap=cm.viridis, color=None, scale_z=True, 
                     view=None, alpha=1.0, shade=False, linewidth=0.1, aa=True):
        n_cells=100
        x, y = np.meshgrid((x_grid[1:]+x_grid[:-1])/2, (y_grid[1:]+y_grid[:-1])/2)
        z = np.nan_to_num(z)
        if scale_z:
            self.ax.set_zlim([np.min(z), 0])
        norm = mpl.colors.Normalize(vmin=np.min(z), vmax=0, clip=False)
        
        # plot the marble
        self.ax.plot([0.], [0.], [0.], marker='o', markersize=15, color='black')

        if color is None:
            self.ax.plot_surface(x, y, z, cmap=cmap, norm=norm, alpha=alpha, shade=shade,
                             rcount=n_cells, ccount=n_cells, linewidth=linewidth, edgecolors='k', antialiased=aa)
        else:
            self.ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=shade, rcount=n_cells, 
                                 ccount=n_cells, linewidth=linewidth, edgecolors='k', antialiased=aa)
        if view == 'top right':
            self.ax.view_init(elev=27, azim=40)

        return self.ax
        
    def add_legend(self, colors, labels):
        patches = [mpl.patches.Patch(color=color, linewidth=0) for color in colors]
        self.ax.legend(patches, labels, fontsize=self.legendFontSize)
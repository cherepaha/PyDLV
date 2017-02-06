import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy import interpolate

class DSPlotter:
    figsize = (14, 8) # in hundreds of pixels

#    x_lim = [-275, 275]
#    y_lim = [-50, 375]
#    z_lim = [-1.5, 0.2]
    n_cells = 30
    
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
#        sns.set_palette('Set3', 10)
     
    def set_axis_params(self, elev=27, azim=130):
        self.ax.xaxis.set_major_locator(MaxNLocator(5))
        self.ax.yaxis.set_major_locator(MaxNLocator(5))
        self.ax.zaxis.set_major_locator(MaxNLocator(1))
        self.ax.set_xlabel(r'x coordinate', fontsize=self.axisLabelFontSize, labelpad=10)
        self.ax.set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize, labelpad=10)
#        self.ax.set_xlim(self.x_lim)
#        self.ax.set_ylim(self.y_lim)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tickLabelFontSize)
        self.ax.view_init(elev, azim) 
        
    def plot_surface(self, x_grid, y_grid, z, cmap=None, color = None, scale_z=True,
                     view='top left', alpha=1.0):
        x, y = np.meshgrid((x_grid[1:]+x_grid[:-1])/2, (y_grid[1:]+y_grid[:-1])/2)
        z = np.nan_to_num(z)
        if scale_z:
            self.ax.set_zlim([np.min(z), 0])
        norm = mpl.colors.Normalize(vmin = np.min(z), vmax = 0, clip=False)
        self.ax.plot([0.], [0.], [0.], marker='o', markersize=15, color = 'black')
        if not color is None:
            self.ax.plot_surface(x, y, z, color = color, norm=norm, alpha=alpha,
                             rstride=1, cstride=1, linewidth=0, antialiased=True)
        elif not cmap is None:
            self.ax.plot_surface(x, y, z, cmap = cmap, norm=norm, alpha=alpha,
                             rstride=1, cstride=1, linewidth=1, antialiased=True)
        else: 
            self.ax.plot_surface(x, y, z, cmap = cm.jet, norm=norm, alpha=alpha,
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
        pass
    
    def overlay_traj(self,trajectories,poly_ds,x_grid,y_grid):
        x=(x_grid[1:]+x_grid[:-1])/2
        y=(y_grid[1:]+y_grid[:-1])/2
        for sim_no, trajectory in trajectories.groupby(level='sim_no'):
#            xx, yy = np.meshgrid(x_grid,y_grid)
            f = interpolate.interp2d(x,y,poly_ds,kind='cubic')
            z = f(trajectory.x.values,trajectory.y.values)
            if trajectory.x.values[-1]>0:
                z= np.diag(z)
            else:
                z=np.diag(np.fliplr(z))
            self.ax.plot(trajectory.x.values, trajectory.y.values,z,label=sim_no)
        
                                 
    def overlay_traj_savefig(self,trajectories,poly_ds,x_grid,y_grid,subj_id):
        plt.ioff()
        x=(x_grid[1:]+x_grid[:-1])/2
        y=(y_grid[1:]+y_grid[:-1])/2
        for sim_no, trajectory in trajectories.groupby(level='sim_no'):
#            xx, yy = np.meshgrid(x_grid,y_grid)
            f = interpolate.interp2d(x,y,poly_ds,kind='cubic')
            z = f(trajectory.x.values,trajectory.y.values)
            if trajectory.x.values[-1]>0:
                z= np.diag(z)
            else:
                z=np.diag(np.fliplr(z))
            self.ax.plot(trajectory.x.values, trajectory.y.values,z,label=sim_no)
        plt.tight_layout()
        plt.savefig('fit_results/by_subject/trajectories/'+subj_id+'_3d'+'.png')
        plt.close()
            
            
                           
    def overlay_traj_single(self,exp_traj,poly_ds,x_grid,y_grid):
        x=(x_grid[1:]+x_grid[:-1])/2
        y=(y_grid[1:]+y_grid[:-1])/2
        f = interpolate.interp2d(x,y,poly_ds,kind='cubic') 
        z = f(exp_traj.x,exp_traj.y)
        if exp_traj.x[-1]>0:
            z= np.diag(z)
        else:
            z=np.diag(np.fliplr(z))

        self.ax.plot(exp_traj.x, exp_traj.y,z,label='experimental trajectory')
        
        
    def overlay_traj_single_control(self,control_traj,poly_ds,x_grid,y_grid):
        x=(x_grid[1:]+x_grid[:-1])/2
        y=(y_grid[1:]+y_grid[:-1])/2
        f = interpolate.interp2d(x,y,poly_ds,kind='cubic') 
        z = f(control_traj[:,0],control_traj[:,1])
        if control_traj[-1,0]>0:
            z= np.diag(z)
        else:
            z=np.diag(np.fliplr(z))

        self.ax.plot(control_traj[:,0], control_traj[:,1],z,label='controlled trajectory')        
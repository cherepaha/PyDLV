from pydlv import data_reader, dl_model_3, dl_generator, dl_plotter
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def plot_dl_variations(dlg, param_names, param_values, colors):
    for i, param_name in enumerate(param_names):
        dlp = dl_plotter.DLPlotter(elev=30, azim=120)
    
        if i>0:
            param_range = np.linspace(param_values[i][0], param_values[i][1], 3)
        else:
            param_range = np.array(param_values[i])
    
        current_params = model.get_baseline_params()
        for n, param_value in enumerate(param_range):
            current_params[i] = param_value
            x_grid, y_grid, dl = dlg.get_model_dl(current_params)
            scale_z = True if n==1 else False
            dlp.plot_surface(x_grid, y_grid, dl, color=colors[n], alpha=0.7, scale_z=scale_z)
        
        title_format = r'$\%s$' if param_name=='tau' else r'$%s$'
        dlp.ax.set_title(title_format % (param_name), loc='left', fontsize=42)        
        dlp.ax.legend(labels, param_range, fontsize=32)
        plt.savefig('figures/param_variations_%s.pdf' % param_name)
            
model = dl_model_3.DLModel3()
dlg = dl_generator.DLGenerator(model)    

dr = data_reader.DataReader()
data = dr.get_processed_data(path='csv/processed_data_high_low.csv')

param_names = ['tau', 'c_{11}', 'c_{21}', 'c_{12}']
k = 0.2
param_values = [[0.04, 0.05, 0.07], [-k*1.0, k*1.0], [-k*1.0, k*1.0], [-k*1.0, k*1.0]]
colors = [cm.inferno(0.2), cm.Greys(0.7), cm.inferno(0.8)]
labels = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]

plot_dl_variations(dlg, param_names, param_values, colors)

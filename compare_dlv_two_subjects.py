import dl_model, decision_landscape_generator, dl_plotter
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

'''
This script demonstrates how, using coefficients from the csv file generated previously,
plot a 3d surface of decision landscape.
'''

def get_fitted_coeffs(method=6):
    data = pd.read_csv('fit_results/by_subject_%i.csv' % (method), header=0)
    data.set_index(['subj_id'], inplace=True)
    return data

def get_subj_dl(model, coeffs, subj_id):
    dlg = decision_landscape_generator.DecisionLandscapeGenerator(model)
    return dlg.get_model_dl(model, coeffs.loc[subj_id][2:9])

coeffs = get_fitted_coeffs()

subj_1 = 9276
subj_2 = 9424

model = dl_model.DLModel()
dlg = decision_landscape_generator.DecisionLandscapeGenerator(model)

subj_1_x, subj_1_y, subj_1_dl = dlg.get_model_dl(model, coeffs.loc[subj_1][2:9])
subj_2_x, subj_2_y, subj_2_dl = dlg.get_model_dl(model, coeffs.loc[subj_2][2:9])

cmap = cm.bwr
colors = [cmap(0.0), cmap(1.0)]

elev, azim = 33, 107
dlp = dl_plotter.DLPlotter(elev, azim)
dlp.plot_surface(subj_1_x, subj_1_y, subj_1_dl, color=colors[0], alpha=0.7)
dlp.plot_surface(subj_2_x, subj_2_y, subj_2_dl, color=colors[1], alpha=0.7)

labels = ['Participant 2', 'Participant 3']
dlp.add_legend(colors, labels)
plt.tight_layout()

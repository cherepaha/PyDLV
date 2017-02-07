import dl_model, decision_landscape_generator, data_reader
import pandas as pd
import os

'''
This script produces a csv file with fitted decision landscape coefficients. The decision landscape
can be fitted either for each trajectory from the input data file individually, or for each subject 
based on that subject's trajectories from the input data file.
'''

def print_subjects_stats(data):
    subjects = data.groupby([data.index.get_level_values(0), 'high_chosen']).high_chosen.count().\
        unstack(fill_value=0)/51
    print(subjects)

def get_fit_parameters(data, methods=[6], mode='by_subject', test_mode=False):
    '''
    The "methods" parameter defines which optimization routines are used to fit the model 
    parameters to the trajectories (see decision_space_generator.py for details). The recommended 
    method is L-BFGS-B. Multiple method codes can be supplied to find best-fit parameters by 
    several methods.
    
    The "mode" defines whether the decision landscape is fitted to each trajectory 
    individually (mode='by_trial') or to all trajectories of each subject (mode='by_subject')
    '''
    # exclude subjects with problematic data
    data = data.drop([2302, 3217], level = 'subj_id')

    # in test mode, to check whether the code works as intended, run parameter fitting 
    # only for part of the data
    if test_mode:
        data = data.loc[[9276, 9424],:]
    
    model = dl_model.DLModel()
    dlg = decision_landscape_generator.DecisionLandscapeGenerator(model)
    
    for method in methods:
        print('Method %i' % method)
        if mode == 'by_subject':
            fit_dl = lambda trajs: dlg.fit_dl_mult_traj(trajs, model, method)    
            coeffs = data.groupby(level='subj_id').apply(fit_dl)
            coeffs.index = coeffs.index.droplevel(1)
            
            # exp_variables is constructed from the initial data file by 
            # dropping t, x, y, vx, vy, ax, ay
            exp_variables = data.groupby(level='subj_id').first(). \
                        drop(data.columns[[0, 1, 2, 12, 13, 14, 15]], axis=1)
        elif mode == 'by_trial':
            fit_dl = lambda traj: dlg.fit_dl_single_traj(traj, model, method)
            coeffs = data.groupby(level = ['subj_id', 'trial_no']).apply(fit_dl)
            coeffs.index = coeffs.index.droplevel(2)
            
            exp_variables = data.groupby(level = ['subj_id', 'trial_no']).first(). \
                        drop(data.columns[[0, 1, 2, 12, 13, 14, 15]], axis=1)
                        
        joined = coeffs.join(exp_variables)
        print('Method %i, median error %f' % (method, joined.error.median()))
        
        if not os.path.exists('fit_results'):
            os.makedirs('fit_results')
        joined.to_csv('fit_results/%s_%i.csv' % (mode, method)) 
        
dr = data_reader.DataReader()
data = dr.get_processed_data()
print_subjects_stats(data)
#get_fit_parameters(data, mode='by_trial', test_mode=True)
get_fit_parameters(data, mode='by_subject', test_mode=True)
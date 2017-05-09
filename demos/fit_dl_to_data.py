import os
from pydlv import data_reader, dl_model_3, dl_generator

'''
This script produces a csv file with fitted decision landscape coefficients. The decision landscape
can be fitted either for each trajectory from the input data file individually, or for each subject 
based on that subject's trajectories from the input data file.
'''

def get_fit_parameters(data, methods=[9], by='subject', csv_path='csv'):
    '''
    The "methods" parameter defines which optimization routines are used to fit the model 
    parameters to the trajectories (see dl_generator.py for details). The recommended 
    methods are 6 (L-BFGS-B) or 9 (SLSQP). Multiple method codes can be supplied to find best-fit 
    parameters by several methods.
    
    The "by" defines whether the decision landscape is fitted to each trial 
    individually (by='trial'), to blocks of trials (by='block'), 
    or to all trajectories of each subject (by='subject')
    '''

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    
    model = dl_model_3.DLModel3()           
    dlg = dl_generator.DLGenerator(model)
    
    for method in methods:
        print('By %s, method %i' % (by, method))
        if by == 'trial':
            fit_dl = lambda traj: dlg.fit_dl_single_traj(traj, method)
            params = data.groupby(level = ['subj_id', 'trial_no']).apply(fit_dl)
            params.index = params.index.droplevel(2)

        elif by == 'block':
            fit_dl = lambda trajs: dlg.fit_dl_mult_traj(trajs, method)        
            params = data.groupby(by=['subj_id', 'block_no']).apply(fit_dl)
            params.index = params.index.droplevel(2)
            
        elif by == 'subject':
            fit_dl = lambda trajs: dlg.fit_dl_mult_traj(trajs,  method)    
            params = data.groupby(level='subj_id').apply(fit_dl)
            params.index = params.index.droplevel(1)
            
        print('By %s, method %i, median error %f' % (by, method, params.error.median()))
        
        file_name = csv_path + '/fit_params_by_%s_method_%i.csv' % (by, method)        
        params.to_csv(file_name)

dr = data_reader.DataReader()        
data = dr.get_processed_data(path='csv/processed_data_high_low.csv')

get_fit_parameters(data, by='trial')
get_fit_parameters(data, by='block')
get_fit_parameters(data, by='subject')
import ds_model, ds_model_alt, decision_space_generator
import pandas as pd

def get_processed_data():
    data = pd.read_csv('../../data/processed_data_high_low.csv', sep=',', header=0)
    data.set_index(['subj_id', 'trial_no'], inplace=True)

    data = data[(data.xflip_count < 2) & (data.chng_mind == False)]    

    return data

def test_single_traj():
    data = get_processed_data()
    traj = data.loc[90, 33]
    model = ds_model.DSModel()
    dsg = decision_space_generator.DecisionSpaceGenerator()
    return dsg.fit_ds_single_traj(traj, model, mode = 12)

def get_fit_coeffs(modes = [1]):
    data = get_processed_data()
#    data = data.drop([2302, 3217], level = 'subj_id')
#    data = data.loc[[9135, 9971],:]
    
#    model = ds_model.DSModel()
    model = ds_model_alt.DSModel()
    dsg = decision_space_generator.DecisionSpaceGenerator()
    
    for mode in modes:
        print('Mode %i' % mode)
        fit_ds = lambda traj: dsg.fit_ds_single_traj(traj, model, mode)
        
        coeffs = data.groupby(level = ['subj_id', 'trial_no']).apply(fit_ds)
        coeffs.index = coeffs.index.droplevel(2)
        
        exp_variables = data.groupby(level = ['subj_id', 'trial_no']).first(). \
                    drop(data.columns[[0, 1, 2, 12, 13, 14, 15]], axis=1)
        joined = coeffs.join(exp_variables)
        print('Mode %i, median error %f' % (mode, joined.error.median()))
        joined.to_csv('fit_results/8_params/%i.csv' % mode)        

#coeffs = test_single_traj()
get_fit_coeffs(modes = [3, 4, 5, 6, 7, 9])
#get_fit_coeffs(modes = [12])
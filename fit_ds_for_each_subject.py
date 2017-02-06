import ds_model, ds_model_alt2, decision_space_generator
import pandas as pd

def get_processed_data():
    data = pd.read_csv('../../data/processed_data_high_low.csv', sep=',', header=0)
    data.set_index(['subj_id', 'trial_no'], inplace=True)

    data = data[(data.xflip_count < 2) & (data.chng_mind == False)]    
    data['high_chosen'] = data.outcome != 5

    return data

def test_single_subject():
    data = get_processed_data()
    trajectories = data.iloc[data.index.get_level_values('subj_id') == 143]
    model = ds_model_alt2.DSModel()
    dsg = decision_space_generator.DecisionSpaceGenerator()
    return dsg.fit_ds_mult_traj(trajectories, model, mode = 11)

def get_fit_coeffs(modes = [1]):
    trajectories = get_processed_data()
#    data = data.drop([2302, 3217], level = 'subj_id')
    trajectories = trajectories.loc[[9424, 9276],:]
#    trajectories = data.iloc[(data.index.get_level_values('subj_id') == 90) | (data.index.get_level_values('subj_id') == 9135)]
    
    model = ds_model_alt2.DSModel()
    dsg = decision_space_generator.DecisionSpaceGenerator()
    
    for mode in modes:
        print('Mode %i' % mode)
        fit_ds = lambda traj: dsg.fit_ds_mult_traj(traj, model, mode)
        
        coeffs = trajectories.groupby(level = 'subj_id').apply(fit_ds)
        coeffs.index = coeffs.index.droplevel(1)
        
        exp_variables = trajectories.groupby(level = ['subj_id', 'trial_no']).first(). \
                    drop(trajectories.columns[[0, 1, 2, 12, 13, 14, 15]], axis=1)
#        joined = coeffs.join(exp_variables)
        print('Mode %i, median error %f' % (mode, coeffs.error.median()))
        coeffs.to_csv('fit_results/by_subject/7_params/%i.csv' % mode)       
    return coeffs

#data = get_processed_data()
#coeffs = test_single_subject()
trajectories = get_processed_data()
subjects = trajectories.groupby([trajectories.index.get_level_values(0), 'high_chosen']).high_chosen.count().unstack(fill_value=0)/51
get_fit_coeffs(modes = [3,4,5,6,7,9])
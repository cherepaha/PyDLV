import decision_space_generator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import odeint
from scipy import integrate
from scipy.optimize import fmin, fmin_powell
import seaborn as sns

trajectory = np.recfromcsv('trajectory_subj_id_143_trial_24.csv', delimiter='\t')

#def eq(par,initial_cond,start_t,end_t,n_steps):
#    t  = np.linspace(start_t, end_t,n_steps)
def eq(par,initial_cond,time_grid):
    def funct(z,t):
        x=z[0]
        y=z[1]
        alpha_0, alpha_1,alpha_2,alpha_3,beta_0,beta_1,beta_2=par
        x_dot = alpha_0 + alpha_1*x + alpha_2*x**2 + alpha_3*x**3
        y_dot = beta_0 + beta_1*y + beta_2*y**2
        return [x_dot, y_dot]
    ds = integrate.odeint(funct,initial_cond,time_grid)
#    return (ds[:,0],ds[:,1],t)
    return (ds,time_grid)

dsg = decision_space_generator.DecisionSpaceGenerator()
p_x, p_y, _, _, _, _ = dsg.polyfit_ds(proc_data, retCoeff=True)
#p_x = p_x[::-1]
#p_y = p_y[::-1]

(alpha_0, alpha_1,alpha_2,alpha_3) = -p_x[::-1][1:]*range(1,5)
(beta_0, beta_1,beta_2) = -p_y[::-1][1:]*range(1,4)

#parameters   
#alpha_0 = 1.0
#alpha_1 = 1.0
#alpha_2 = 1.0
#alpha_3 = 1.0
#beta_0 = 1.0
#beta_1 = 1.0
#beta_2 = 1.0

params=(alpha_0,alpha_1,alpha_2,alpha_3,beta_0,beta_1,beta_2)

exp_t=trajectory.t
XY_data=np.array([trajectory.x, trajectory.y]).T

# initial conditions
x_0 = trajectory.x[0]
y_0 = trajectory.y[0]
ic = [x_0, y_0]
 
sol,_=eq(params,ic,exp_t)
 

######### trying to fit dynamical system parameters
#X=trajectory.x
#Y=trajectory.y
# model steps
#---------------------------------------------------
start_time=exp_t[0]
end_time=exp_t[-1]
#intervals=100
#mt=np.linspace(start_time,end_time,intervals)
 
# MAGIC: model index to compare to data
#----------------------------------------------------
#findindex=lambda x:np.where(mt>=x)[0][0]
#mindex=map(findindex,T)

def score(parms):
    sol,_=eq(parms,ic,exp_t)
#    time_matched_sol = sol[mindex,:]
    err=lambda data,model:np.sqrt((data[0]-model[0])**2+(data[1]-model[1])**2).sum()
    return err(XY_data,sol)

fit_score=score(params)
answ=fmin(score,(params),full_output=1,maxiter=10000000)
bestrates=answ[0]
bestscore=answ[1]
alpha_0,alpha_1,alpha_2,alpha_3,beta_0,beta_1,beta_2=answ[0]
newparams=(alpha_0,alpha_1,alpha_2,alpha_3,beta_0,beta_1,beta_2)

sol,_=eq(newparams,ic,exp_t)

plt.figure()
plt.plot(trajectory.x, trajectory.y)
plt.plot(sol[0], sol[1])

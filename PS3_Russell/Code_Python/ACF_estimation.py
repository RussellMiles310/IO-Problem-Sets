# -*- coding: utf-8 -*-
"""
ACF Estimation
"""

from source_functions import *

def ACF_estimation(df):

#=== options =================================================================#
    degree= 3 #polynomial fit degree
    
#=== Fit Phi =================================================================#    
    
    xvars = df[['k', 'l', 'm']].to_numpy()
    y = df[['y']].to_numpy()
    X_poly = poly_design_matrix(xvars, degree)
    
    Phi = regress(y, X_poly)[1]
    
    df["Phi"] = Phi
    #Add into the dataframe
    df['Phiprev'] = df.groupby('firm_id')['Phi'].shift(1)
    
    
#=== Drop NaNs after creating the lagged Phi. Define GMM arguments ===========#    
    df_nonans = df.dropna()
    #Get all the variables out of the dataframe -- This allows me to use Autograd
    y = df_nonans['y'].to_numpy() 
    k = df_nonans['k'].to_numpy()  
    l = df_nonans['l'].to_numpy()  
    Phi = df_nonans['Phi'].to_numpy() 
    kprev = df_nonans['kprev'].to_numpy()  
    lprev = df_nonans['lprev'].to_numpy() 
    Phiprev = df_nonans['Phiprev'].to_numpy() 
    #Run GMM
    #Initial guess for parameters beta_k, beta_l
    theta0 = np.array([1,1])
    #Weight matrix -- use the identity for now. 
    W0 = np.eye(2)
    #(2) Get matrix of variables used in exogeneity restrictions
    Vex = moment_ex_restrictions_ACF(k, lprev)

#=== Run GMM =================================================================#    

    autogradient = grad(gmm_obj_ACF)

    gmm_args = (y, k, l, kprev, lprev, Phi, Phiprev, Vex, W0)
    
    tolerance = 1e-25
    
    #theta_results = opt.minimize(gmm_obj_ACF, theta0, args=gmm_args,
    #                        tol=tolerance, method='Nelder-Mead', options={'maxiter': 10000})
    
    theta_results_grad = opt.minimize(gmm_obj_ACF, theta0, args=gmm_args,
                           tol=tolerance, jac=autogradient, method='L-BFGS-B',                              
                            options={'ftol': 1e-16, 'gtol': 1e-16, 'maxiter': 10000})
        
    theta=theta_results_grad.x
    #Get the slope, rho. It's the slope of the regression used to find the moments. 
    rho = moment_error_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev)[1]
    
    print("The gradient at the optimum is: ", autogradient(theta, y, k, l, kprev, lprev, Phi, Phiprev, Vex, W0))
    print("The GMM error using the gradient is:", gmm_obj_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev, Vex, W0))
    print("The estimates using autograd: [beta_k, beta_l] = ", theta)
    print("The slope of the AR(1) of productivity is: rho = ", rho)

#=============================================================================#    

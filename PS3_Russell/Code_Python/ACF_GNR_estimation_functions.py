# -*- coding: utf-8 -*-
"""
ACF Estimation
"""

from source_functions import *

def ACF_estimation(df, theta0, print_results = 0):

#=== options =================================================================#
    degree= 3 #polynomial fit degree
    #theta0 = np.array([1,1])     #Initial guess for parameters beta_k, beta_l
    W0 = np.eye(2)     #Weight matrix -- use the identity for now. 

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
                            options={'ftol': tolerance, 'gtol': tolerance, 'maxiter': 20000})
        
    theta=theta_results_grad.x
    #Get the slope, rho. It's the slope of the regression used to find the moments. 
    rho = moment_error_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev)[1][1]
    
    if print_results == 1:
        print("The gradient at the optimum is: ", autogradient(theta, y, k, l, kprev, lprev, Phi, Phiprev, Vex, W0))
        print("The GMM error using the gradient is:", gmm_obj_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev, Vex, W0))
        print("The estimates using autograd: [beta_k, beta_l] = ", theta)
        print("The slope of the AR(1) of productivity is: rho = ", rho)
    
    
    #Calculate omega and beta0
    xi, Rho, b0_plus_omega, b0_plus_omega_prev = moment_error_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev)

    omegaprev = (b0_plus_omega-b0_plus_omega_prev - Rho[0] - xi)/(Rho[1]-1)
    omega = Rho[0] + Rho[1]*omegaprev + xi
    
    Eomega = np.mean(omega)
    
    Ebeta0 = np.mean(b0_plus_omega-omega)
    
    results_coefficients = np.array([Ebeta0, theta[0], theta[1], rho, Eomega])
    results_convergence = gmm_obj_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev, Vex, W0)
    
    #df_nonans['omega'] = omega
    #df_nonans['omegaprev'] = omegaprev
    
    return results_coefficients, results_convergence

#=============================================================================#    
def GNR_estimation(df, initial_guesses, print_results = 0):

    
    alpha0, gammaprime0 = initial_guesses

#=== options =================================================================#
    degree= 2
    degree_omega = 2
    
#=== Fit D_{jt} ==============================================================#    

    #Make the polynomial design matrix
    xvars = df[['k', 'l', 'm']].to_numpy()
    s = df[['s']].to_numpy()
    X_poly_D = poly_design_matrix(xvars, degree)
    #calculate the gradient of the objective function using AutoGrad
    autogradient_nlls = grad(nlls_share_obj)
    autohessian_nlls = hessian(nlls_share_obj)
    #initial guess
    #gammaprime0 = np.ones(X_poly_D.shape[1])/2

    #minimize to fit the coefficients gammaprime 
    #Enforce that X@gamma is nonnegative, otherwise we get negative values in the log
    nonnegative_b = {'type': 'ineq', 'fun': lambda b: (X_poly_D@b)}
    
    gammaprime_results = opt.minimize(nlls_share_obj, gammaprime0, args=(X_poly_D, s),
                                      constraints = [nonnegative_b],
                           tol=1e-12, jac=autogradient_nlls, hess = autohessian_nlls, method='trust-constr'
    )
    
    #print("The error is:",  gammaprime_results.fun)
    #print("The gradient is:",  gammaprime_results.grad)
    #print("The coefficients in the degree-1 fit are:",  gammaprime_results.x)
    
    #shat = np.log(X_poly_D@gammaprime_results.x)
    gammaprime = gammaprime_results.x
    #Get Dhat, the elasticities
    df['Dhat'] = X_poly_D@gammaprime
    #Back out the residuals, epsilons
    df['epsilonhat'] = np.log(df['Dhat']) - df['s']
    # mean of epsilon is 1e-12 --- good sign
    CurlyEhat = (np.mean(np.exp(df['epsilonhat'])))
    #The theoretial guess for CurlyEhat given epsilon ~ N(0, sigma^2) is very close to the actual curlyEhat, 
    #suggesting the epsilons are approximately normally distributed. 
    #It follows from the math above that ...
    gamma = gammaprime/CurlyEhat
    df['df_dm'] = X_poly_D@gamma
 
#=== Evaluate the integral, CurlyD ===========================================#   
#Then calculate some more objects needed for GMM estimation. 
    #Get the design matrix associated with the integral of the polynomial
    X_poly_D_integral =  poly_integral_design_matrix(xvars, degree, w_r_t = 2)
    #Evaluate it to get curlyD, which is the integral of the log elasticities
    df['CurlyD'] = X_poly_D_integral@gamma 
    #from here, get CurlyY
    df['CurlyY'] = df['y'] - df['epsilonhat'] - df['CurlyD']
    df['CurlyYprev'] = df.groupby('firm_id')['CurlyY'].shift(1)
    
    #Now, drop all NaNs
    df_nonans = df.dropna().copy()
    
    xvars_omega = df_nonans[["k", "l"]].to_numpy()
    xvars_prev_omega = df_nonans[["kprev", "lprev"]].to_numpy()
    
    #This polynomial fit has NO INTERCEPT. Even if we wanted an intercept it would not be identified because we end up taking first differences of omega. 
    X_poly_omega = poly_design_matrix(xvars_omega, degree_omega)[:, 1:]
    Xprev_poly_omega = poly_design_matrix(xvars_prev_omega, degree_omega)[:, 1:]
    
    #Previous CurlyY
    CurlyY = df_nonans['CurlyY'].to_numpy()
    CurlyYprev = df_nonans['CurlyYprev'].to_numpy()
    
    #alpha0 = np.ones(X_poly_omega.shape[1])/2
    W0 = np.eye(len(alpha0))
    
#=== Run the GMM estimation  =================================================#    
    
    args_GNR = (X_poly_omega, Xprev_poly_omega, CurlyY, CurlyYprev, W0)
    
    tolerance = 1e-24
    
    gmm_results_GNR = opt.minimize(gmm_obj_fcn_GNR, alpha0, args=args_GNR,
                           tol=1e-24, jac=autogradient_GNR, method='L-BFGS-B', 
                           options={'ftol': tolerance, 'gtol': tolerance, 'maxiter': 20000}
    )
    
    alpha = gmm_results_GNR.x
    delta, eta = gmm_stage2_error_GNR(alpha, X_poly_omega, Xprev_poly_omega, CurlyY, CurlyYprev)[1:3]
    
    #Calculate the omegas
    df_nonans['ConstantC'] = X_poly_omega@alpha
    df_nonans['omega'] = df_nonans['ConstantC'] + df_nonans['CurlyY']
    
    Eomega = np.mean(df_nonans['omega'])
    Edf_dm = np.mean(df_nonans['df_dm'])
    
    
    #Assuming Cobb-Douglas, we can get elasticities with OLS
    df_nonans['f'] = df_nonans['CurlyD'] - df_nonans['ConstantC']
    #df_nonans['f_plus_epsilon'] = df_nonans['y'] - df_nonans['omega']
    f = df_nonans['f']
    
    #Run OLS
    klm = df_nonans[['k', 'l', 'm']]
    Xklm = np.hstack((np.ones((klm.shape[0],1)), klm.to_numpy()))
    fbeta_cobbdouglas, _, _ = regress(f, Xklm)
    
    if print_results == 1:
        print("The error is:",  gmm_results_GNR.fun)
        print("The gradient is:",  gmm_results_GNR.jac)
        print("The coefficients for the integration constant [alpha] are:",  gmm_results_GNR.x)
        print("The coefficients for productivity omega [delta] are:",  delta)
        print("the average productivity [omega] is:", Eomega)
        print("the average elasticity [df/dm] is:", Edf_dm)
        print("----Assuming Cobb-Douglas----")
        print("[beta_0, beta_k, beta_l, beta_m] = ", fbeta_cobbdouglas.flatten())
    
    results_params = np.concatenate((fbeta_cobbdouglas.flatten(), [Edf_dm], [Eomega]))
        
    results_convergence = gmm_results_GNR.fun
    
    return results_params, results_convergence, alpha, gammaprime
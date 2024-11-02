# -*- coding: utf-8 -*-
"""
Main script to create point estimates and bootstrapped standard errors for ACF and GNR
For bootstrapping, we sample FIRMS with replacement in order to preserve the time series properties of the data. 

Some things are hard-coded within the estimation functions -- without time constraints, we would code this more elegantly. 
"""
from ACF_GNR_estimation_functions import *


#==== Options ================================================================#
#Number of bootstrap samples
np.random.seed(9)
n_boot_samples = 100

#==== load_data ==============================================================#

df_ACF = load_data("ACF")
df_GNR = load_data("GNR")

#==== point estimate for ACF ==================================================#
print("---------------------------------------------------")
print("-----------ACF: Getting Point Estimates -----------")
print("---------------------------------------------------")
theta0 = np.array([1,1])/2 #initial guess for ACF parameters
coeffs_ACF, convergence = ACF_estimation(df_ACF, theta0, print_results=1)
theta_ACF = coeffs_ACF[1:3]

#==== point estimate for GNR ==================================================#
#initial guesses for GNR parameters. 
print("---------------------------------------------------")
print("-----------GNR: Getting Point Estimates -----------")
print("---------------------------------------------------")

alpha0 = np.ones(5)/2 #This is the required size to have coefficeints for k, l, kl, k**2, l**2. Need to change if the degree is changed
gammaprime0 = np.ones(10)/2  #Also needs to change if the degree is changed
initial_guesses0 = (alpha0, gammaprime0)

results_params_GNR, results_convergence_GNR, alpha_GNR, gammaprime_GNR = GNR_estimation(df_GNR, initial_guesses0, print_results = 1)
initial_guesses_GNR = (alpha_GNR, gammaprime_GNR)

#==== Run bootstrap for ACF, save results ====================================#
print("---------------------------------------------------")
print("-----------ACF: Bootstrapping Standard Errors------")
print("---------------------------------------------------")
bootstrap_results_ACF, convergence_ACF = bootstrap(ACF_estimation, theta_ACF, df_ACF, n_boot_samples)

#Summarize array
ACF_row_names = np.array(["beta_0", "beta_k", "beta_l", "rho", "Eomega", "gmm_error"])
ACF_summary = summarize_array(coeffs_ACF, bootstrap_results_ACF, ACF_row_names[:-1])

boot_full_ACF = pd.DataFrame(np.hstack((bootstrap_results_ACF, convergence_ACF)), columns = ACF_row_names) 

#Save to CSV
ACF_summary.to_csv("../Results/summary_stats_ACF.csv")
boot_full_ACF.to_csv("../Results/full_bootstrap_ACF.csv")

#==== Run bootstrap for GNR, save results ====================================#
#Use the initial condition from the true data to improve speed and convergence. 
print("---------------------------------------------------")
print("-----------GNR: Bootstrapping Standard Errors------")
print("---------------------------------------------------")
bootstrap_results_GNR, convergence_GNR = bootstrap(GNR_estimation, initial_guesses_GNR, df_GNR, n_boot_samples, columns = 6)

#Summarize array
GNR_row_names = np.array(["beta_0_cd", "beta_k_cd", "beta_l_cd", "beta_m_cd", "Edf_dm", "Eomega", "gmm_error"])
GNR_summary = summarize_array(results_params_GNR, bootstrap_results_GNR, GNR_row_names[:-1])

boot_full_GNR = pd.DataFrame(np.hstack((bootstrap_results_GNR, convergence_GNR)), columns = GNR_row_names) 

#Save to CSV
GNR_summary.to_csv("../Results/summary_stats_GNR.csv")
boot_full_GNR.to_csv("../Results/full_bootstrap_GNR.csv")

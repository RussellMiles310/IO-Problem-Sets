"""
Source functions for GNR and ACF estimations
Detailed Jupyter notebooks describing the logic of these functions are attached. 
"""

#Source functions
import autograd.numpy as np
from autograd import grad, hessian
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from itertools import combinations_with_replacement, chain #used for constructing polynomials
from numba import jit



#==== Used in ACF and GNR ====================================================#

#Load the data
def load_data(model):
    
    filename = "../PS3_data_changedtoxlsx.xlsx"
    df0 = pd.read_excel(filename)
    #Remove missing materials columns
    df = df0[['year', 'firm_id', 'X03', 'X04', 'X05', 'X16', 'X40', 'X43', 'X44', 'X45', 'X49']]
    #new_names = ["year", "firm_id", "obs", "ly", "s01", "s02", "lc", "ll", "lm"]
    new_names = ["t", "firm_id", "y_gross", "s01", "s02", "s13", "k", "l", "m", 'py', 'pm']
    df.columns = new_names
    #Drop missing materials data
    df=df[df['m']!=0]
    #Keep industry 1 only
    df=df[df['s13']==1]
    
    if model == "ACF":
        #Creating value-added y
        df['y'] = np.log(np.exp(df['y_gross'] + df['py']) - np.exp(df['m'] + df['pm']))
    elif model == "GNR":
        #in GNR, we simply use gross y
        df['y'] = df['y_gross']
        df['s'] = df['pm']+df['m'] - df['py'] - df['y']
    else: 
        print("Please enter the string ACF or GNR" )

    #Creating lagged variables
    df = df.sort_values(by=['firm_id', 't'])
    df['kprev'] = df.groupby('firm_id')['k'].shift(1)
    df['lprev'] = df.groupby('firm_id')['l'].shift(1)
    df['mprev'] = df.groupby('firm_id')['m'].shift(1)
    
    return df



#Creates an iterator of tuples, useful for constructing polynomial regression design matrices. 
def poly_terms(n_features, degree):
    #This thing creates an iterator structure of tuples, used to create polynomial interaction terms. 
    #It looks something like this: (0,), (1,), (2,), (0, 0), (0, 1) 
    polynomial_terms = chain(
        *(combinations_with_replacement(range(n_features), d) for d in range(1, degree+1))
    )
    return(polynomial_terms)

#Contructs a polynomial regression design matrix. 
def poly_design_matrix(xvars, degree):
    if xvars.ndim == 1:
        xvars = xvars.reshape(1, -1)
    # Get the number of samples (n) and number of features (m) from X
    n_samples, n_features = xvars.shape
    # Start with a column of ones for the intercept term
    X_poly = np.ones((n_samples, 1))
    #Create iterator used to construct polynomial terms
    polynomial_terms = poly_terms(n_features, degree)
    # Generate polynomial terms and interaction terms up to 4th degree
    for terms in  polynomial_terms:  # For degrees 1 to 4
            #print(terms)
            X_poly = np.hstack((X_poly, np.prod(xvars[:, terms], axis=1).reshape(-1, 1)))
    # Compute the coefficients using the normal equation: beta = (X.T * X)^(-1) * X.T * y
    return X_poly

#Runs a regression 
def regress(y, X):
    beta = np.linalg.solve(X.T@X, X.T@y)
    yhat = X@beta
    resids = y-yhat
    return beta, yhat, resids

#==== Used in ACF only =======================================================#
#Calculates the error term, h(theta, y, k, l)
def moment_error_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev):
    #get the innovations to omega
    beta_k = theta[0]
    beta_l = theta[1]
    b0_plus_omega = Phi - beta_k*k - beta_l*l 
    b0_plus_omega_prev = Phiprev - beta_k*kprev - beta_l*lprev 
    #Regress them to get the innovations
    yvar = b0_plus_omega#.reshape(-1, 1)
    xvar = b0_plus_omega_prev.reshape(-1, 1)
    #Degree of the Omega polynomial
    omega_degree = 1
    Xdesign = poly_design_matrix(xvar, omega_degree)
    #coeffs will contain rho, the AR(1) slope of productivity
    #b0_plus_omega_hat is the predicted value. 
    coeffs, b0_plus_omega_hat = regress(yvar, Xdesign)[:2]
    #Get residual
    xi = b0_plus_omega -  b0_plus_omega_hat 
    return xi, coeffs, b0_plus_omega, b0_plus_omega_prev

def moment_ex_restrictions_ACF(k, lprev):
    #Moment conditions include exogeneity restrictions for 1, k_{it}, l_{it-1}, and Phi. 
    #Put them all in one matrix for easy access, called Vexc (short for vectors for exogeneity restrictions)
    #Replace all nans with zeros -- this is ok, because we're just taking a dot product over each row of this matrix, and want to remove the nans
    Vex = np.vstack([
        k, 
        lprev])
    return Vex

def gmm_obj_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev, Vex, W):
    #Arguments
    #Get the vector h(theta, y, k, l)
    xi = moment_error_ACF(theta, y, k, l, kprev, lprev, Phi, Phiprev)[0]
    #Calculate the "error" -- exogenous terms (dotproduct) h(theta, y, k, l)
    err = (Vex@xi)/len(xi)
    #Calculate the weighted sum of the error using the weight matrix, W
    obj = err.T@W@err
    return obj

#==== Used in GNR only =======================================================#
#Function for finding the nonlinear least squares objective function 
#This is used to fit the regression of log shares on the log of the polynomial approximation of D_{jt}
def nlls_share_obj(gamma, X_poly, s):
    #gamma is the vector of coefficients
    #X_poly is the design matrix containing all of the polynomial coefficients
    Dhat = X_poly@gamma
    #Evaluate the objective -- the sum of squared residuals
    obj = np.sum((s.flatten() - np.log(Dhat))**2) #/(X_poly.shape[0])
    return obj

#Used to integrate the polynomial, in order to get CurlyD.
#by default,integrate with respect to to xvars[2] = m
def poly_integral_design_matrix(xvars, degree, w_r_t = 2):
    #Get number of observations (n) and number of independent variables (k)
    if xvars.ndim == 1:
        xvars = xvars.reshape(1, -1)
    # Get the number of samples (n) and number of features (m) from X
    n_samples, n_features = xvars.shape
    # Start with a column of ones for the intercept term
    X_poly_integral0 = np.ones((n_samples, 1))
    #Create iterator used to construct polynomial terms
    polynomial_terms = poly_terms(n_features, degree)
    # Generate polynomial terms and interaction terms up to 4th degree
    for terms in  polynomial_terms:  # For degrees 1 to 4
            integration_divisor = terms.count(w_r_t) + 1 #count the number of xvars[2] (i.e. m) appearing in the term and add 1. 
                                                         #Divide the column by that term to campute the "integration scalar"
            xcolumn = np.prod(xvars[:, terms], axis=1).reshape(-1, 1) / integration_divisor
            X_poly_integral0 = np.hstack((X_poly_integral0, xcolumn))
    #Elementwise-multiply all columns in the resulting matrix by m
    X_poly_integral = X_poly_integral0 * xvars[:,w_r_t].reshape(xvars.shape[0],1)
    return X_poly_integral

#Moment error function
def gmm_stage2_error_GNR(alpha, X_poly_omega, Xprev_poly_omega, CurlyY, CurlyYprev):
    #Given alpha, the previous omega is Curly Y + the integration constant curlyC, which is a polynomial fit on lagged k and l
    #Note that there's NO INTERCEPT in this polynomial fit
    #Even if we included an intercept, it would not be identified. 
    omegaprev = CurlyYprev + Xprev_poly_omega@alpha
    #Then, calculate current omega = curlyY
    omega = CurlyY + X_poly_omega@alpha
    #Regress omega on omegaprev
    degree_omega = 2
    #xvars
    Xo = poly_design_matrix(omegaprev.reshape(-1, 1), degree_omega)
    #Fit the regression omegaprev ~ polynomial(omega)
    #delta is the coefficients, eta are the residuals
    delta, _, eta = regress(omega, Xo)
    #calculate the moment errors
    #The moments are simply the polynomial terms in poly(alpha) used to approximate the constant of integration. 
    moment_error = (X_poly_omega.T @ eta)/len(eta)
    return moment_error, delta, eta

def gmm_obj_fcn_GNR(alpha, X_poly_omega, Xprev_poly_omega, CurlyY, CurlyYprev, W):
    #Get the moment errors
    moment_error = gmm_stage2_error_GNR(alpha, X_poly_omega, Xprev_poly_omega, CurlyY, CurlyYprev)[0]
    #caluclate GMM error objective function
    obj = moment_error.T@W@moment_error
    return obj

#Define autogradient
autogradient_GNR = grad(gmm_obj_fcn_GNR)



#==== functions for bootstrapping ============================================#


def bootstrap(func, param0, df, n_samples = 1000, columns=5):
    # Store results
    coefficients = np.zeros((n_samples, columns))  # 3 coefficients
    convergence = np.zeros((n_samples, 1))
    
    list_df_boot = bootstrap_sample_panel(df, n_samples)
    
    
    # Perform bootstrap sampling
    for i in range(n_samples):
        # Sample with replacement        
        # Get coefficients from the provided function
        printflag = 0
        param0_temp = param0
        for tries in range(50):
            coefs, conv = func(list_df_boot[i], param0_temp)[:2]
            if conv < 1e-10:
                if printflag == 1:
                    print("convergence succeeded on sample", i)
                break
            else:  #Run again if no convergence (occurs rarely)
                print("convergence failed on sample", i, "; trying new initial guess.")
                printflag = 1
                perturb = np.random.uniform(0.3, 3)
                if isinstance(param0_temp, tuple):
                    a, b = param0
                    param0_temp = (a*perturb, b)
                else:
                    param0_temp = param0*perturb

                
        coefficients[i,:] =coefs
        convergence[i] =conv

    # Return the bootstrap results
    return coefficients, convergence

# Bootstrap sampling function
def bootstrap_sample_panel(data, n_samples, id_col='firm_id'):
    # Create an empty list to store bootstrap samples
    bootstrap_samples = []
    original_firms = data[id_col].unique()
    num_firms = len(original_firms)

    for _ in range(n_samples):
        # Sample 'firm_id's with replacement
        sampled_firms = np.random.choice(original_firms, size=num_firms, replace=True)

        # Initialize a list to store the sampled data with unique firm IDs
        sample_data_list = []

        for i, firm in enumerate(sampled_firms):
            # Extract the firm data block
            firm_data = data[data[id_col] == firm].copy()

            # Assign a new unique firm ID by offsetting with the current sample index
            firm_data[id_col] = f"{firm}_{i}"  # e.g., '1_0', '2_1', etc.

            # Append to the list for this sample
            sample_data_list.append(firm_data)

        # Concatenate all firm blocks in this sample
        sample_data = pd.concat(sample_data_list).reset_index(drop=True)
        
        # Append to list of samples
        bootstrap_samples.append(sample_data)

    return bootstrap_samples

def summarize_array(point_estimate, arr, row_names):
    # Ensure arr is a NumPy array
    arr = np.asarray(arr)
    
    # Validate row_names length
    if len(row_names) != arr.shape[1]:
        raise ValueError("The length of row_names must match the number of columns in the array.")
    
    # Compute summary statistics
    mean = np.mean(arr, axis=0)
    p2_5 = np.percentile(arr, 2.5, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    median = np.median(arr, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p97_5 = np.percentile(arr, 97.5, axis=0)
    std_error = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])

    # Create a DataFrame to organize results
    summary_df = pd.DataFrame({
        'Point Estimate': point_estimate,
        'Bootstrap Mean': mean,
        '2.5th Percentile': p2_5,
        '25th Percentile': p25,
        'Median': median,
        '75th Percentile': p75,
        '97.5th Percentile': p97_5,
        'Standard Error': std_error
    }, index=row_names)

    return summary_df
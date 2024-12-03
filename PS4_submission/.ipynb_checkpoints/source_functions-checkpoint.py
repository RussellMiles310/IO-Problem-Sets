# -*- coding: utf-8 -*-
"""
Source functions
"""
import pandas as pd
import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
from jax import grad, jacobian, hessian, config, jit, lax, jacfwd
import jax.numpy as jnp
import warnings
from functools import partial


config.update("jax_enable_x64", True)


#=============================================================================#
# Arguments common among all functions
#=============================================================================#
# params: parameter vector [sigma, etas, deltas]
# M: number of markets
# J: number of products per market
# N_instruments: number of instruments/moment conditions
# N: Number of consumers in the random draw. 


#=============================================================================#
# s: Function to calculate market shares
#=============================================================================#
#Arguments:
# nus_on_prices: The random component of utility from prices. 
@partial(jit, static_argnums=(2,))
def s(params, nus_on_prices, MJN):
    
    M, J, N_instruments, N = MJN
    sigma = params[0]
    
    # Use lax.dynamic_slice for dynamic slicing
    deltas_start = N_instruments + 1
    deltas = lax.dynamic_slice(params, (deltas_start,), (params.shape[0] - deltas_start,)).reshape(-1, 1)
    
    # Compute utilities
    utilities = deltas - sigma * nus_on_prices  # Shape: (M*J, N)
    
    # Reshape utilities for markets and products
    utilities_reshaped = utilities.reshape(M, J, N)  # Shape: (M, J, N)
    
    # Compute the stabilization constant (max utility per market per individual)
    max_utilities = jnp.max(utilities_reshaped, axis=1, keepdims=True)  # Shape: (M, 1, N)
    
    # Stabilized exponentials
    exp_utilities = jnp.exp(utilities_reshaped - max_utilities)  # Shape: (M, J, N)
    
    # Adjust the "outside option" (1 becomes exp(-max_utilities))
    outside_option = jnp.exp(-max_utilities)  # Shape: (M, 1, N)
    
    # Compute the stabilized denominator
    sum_exp_utilities = outside_option + exp_utilities.sum(axis=1, keepdims=True)  # Shape: (M, 1, N)
    
    # Compute shares
    shares = exp_utilities / sum_exp_utilities  # Shape: (M, J, N)
    
    # Average across individuals
    avg_shares = shares.mean(axis=2)  # Shape: (M, J)
    
    # Flatten output to match the original function's shape
    return avg_shares.flatten()

#=============================================================================#
# solve_init_deltas: function to get a feasible initial guess for deltas. 
#=============================================================================#
# shares: the vector of true market shares
def solve_init_deltas(params, shares, nus_on_prices, MJN):
    
    M, J, N_instruments, N = MJN
    
    constraint_func = lambda x: (s(x, nus_on_prices, MJN).reshape(J*M, -1) - shares).reshape(-1)
    constraint_jac = jacobian(constraint_func)

    # Define the constraint dictionary
    constraints = {
        'type': 'eq',
        'fun': constraint_func,
        'jac': constraint_jac
    }

    # Perform optimization
    result = minimize(
        fun=lambda x: 0,
        x0=params,
        method='SLSQP',
        constraints=constraints
    )
    
    # Return results
    if result.success:
        return result
    else:
        print("Optimization failed:", result.message)
        print("Returning the original passed parameters.")
        return params

#=============================================================================#
# blp_instruments_all: gets matrix of instruments used in moment conditions 
#=============================================================================#
def blp_instruments_all(X, W_costs, Z_costs, prices, MJN):
    """
    Computes the matrix of instruments for all (j, m) pairs in a vectorized manner.
    
    Parameters:
    ----------
    X : jnp.ndarray
        Input matrix of shape (J * M, features).
        
    Returns:
    -------
    instruments : jnp.ndarray
        Matrix of instruments for all (j, m), shape (J * M, 6).
    """
    M, J, _, N = MJN
    
    # Reshape X into (M, J, features)
    # Note: for instruments, use only the nonconstant product characteristics.
    X_reshaped = X[:, 1:].reshape(M, J, -1)  # Shape: (M, J, features)

    # First two elements: Features of product j in market m
    X_jm = X_reshaped  # Shape: (M, J, features)

    # Next two elements: Sum of product j's features in all other markets
    X_j_sum = jnp.sum(X_reshaped, axis=0, keepdims=True) - X_reshaped  # Shape: (M, J, features)

    # Next two elements: Sum of all other products' features in the same market
    X_m_sum = jnp.sum(X_reshaped, axis=1, keepdims=True) - X_reshaped  # Shape: (M, J, features)

    # Next element: W (in the marginal cost function)
    W_costs_reshaped = W_costs.reshape(M, J, -1)
    
    # Next element: W (in the marginal cost function)
    Z_costs_reshaped = Z_costs.reshape(M, J, -1)

    # Final element: Prices
    prices_reshaped = prices.reshape(M, J, -1)
    
    # Concatenate results along the last dimension and add a column of ones
    instruments = jnp.concatenate([
        jnp.ones((M, J, 1)),
        X_jm,
        X_j_sum,
        X_m_sum,
        W_costs_reshaped,
        prices_reshaped, 
        Z_costs_reshaped
    ], axis=-1)  # Shape: (M, J, 7)
    
    # Reshape back to (J * M, 6)
    return instruments.reshape(J * M, -1)


#=============================================================================#
# blp_moment: gets matrix of instruments used in moment conditions 
#=============================================================================#
# Z: matrix of instruments used to calculate moment conditions
# Az: Annihilator matrix, used to recover xis from deltas. 
def blp_moment(params, Z, Az, MJN):
    """
    Computes the BLP moment vector using vectorized instruments.
    
    Parameters:
    ----------
    params : array-like
        Model parameters.
    X : jnp.ndarray
        Input data matrix of shape (J * M, features).
        
    Returns:
    -------
    sum_vec : jnp.ndarray
        The moment vector divided by the number of market and products, shape (instrument_features,).
    """
    M, J, N_instruments, N = MJN

    deltas = params[1+N_instruments:].reshape(-1, 1)  # Shape: (J * M, 1)
    
    xis = Az @ deltas # Use the annihilator matrix to recover xi
    sum_vec = jnp.sum(xis*Z, axis=0)  # Shape: (instrument_features,)
    return (sum_vec / (J*M))


#=============================================================================#
# blp_moment: gets matrix of instruments used in moment conditions 
#=============================================================================#
# Z: matrix of instruments used to calculate moment conditions
# Az: Annihilator matrix for the demand-side, used to recover xis from deltas. 
# Xs: matrix of supply-side regressors: [1, wcost, zcost]
# As: Annihilator matrix for the suppply side, used to recover the marginal cost residual, omega. 
def blp_moment_joint(params, X, Z, Az, M_iv_est, Xs, As, prices, shares, nus, nus_on_prices, MJN, ownership):
    """
    Computes the BLP moment vector using vectorized instruments.
    
    Parameters:
    ----------
    params : array-like
        Model parameters.
    X : jnp.ndarray
        Input data matrix of shape (J * M, features).
        
    Returns:
    -------
    sum_vec : jnp.ndarray
        The moment vector divided by the number of market and products, shape (instrument_features,).
    """
    ###### First: Demand-side moments
    M, J, N_instruments, N = MJN
    deltas = params[1+N_instruments:].reshape(-1, 1)  # Shape: (J * M, 1)
    xis = Az @ deltas # Use the annihilator matrix to recover xi
    moment_demand_side = jnp.sum(xis*Z, axis=0)  # Shape: (instrument_features,)
    
    ###### Next: Supply-side moments
    
    if not ownership.any(): ## I am coding perfect competition as an ownership matrix of zeros. 
        mc=prices ### Perfect competition case. 
    else:
        mc = calculate_marginal_costs(params, ownership, xis, X, M_iv_est, prices, nus, nus_on_prices, MJN)
    # Find the residual of the marginal cost equation
    omegas = As @ mc
    moment_supply_side = jnp.sum(omegas*Xs, axis=0)  # Shape: (instrument_features,)
    #Put the moments together
    moments_all = jnp.concatenate([moment_demand_side, moment_supply_side], axis = 0)
    return (moments_all / (J*M))


#=============================================================================#
# objective_mpec
#=============================================================================#
def objective_mpec(params, W, MJN):
    M, J, N_instruments, N = MJN
    eta = jnp.array(params[1:1+N_instruments])
    out = eta.T @ W @ eta
    return out
#=============================================================================#
# constraint_g
#=============================================================================#
def constraint_g(params, Z, Az, MJN):
    _, _, N_instruments, _ = MJN
    g_xi = blp_moment(params, Z, Az, MJN)
    eta = params[1:1+N_instruments]
    return g_xi - eta


#=============================================================================#
# constraint_g_joint
#=============================================================================#
#@partial(jit, static_argnums=(10,))
def constraint_g_joint(params, X, Z, Az, M_iv_est, Xs, As, prices, shares, nus, nus_on_prices, MJN, ownership):
    _, _, N_instruments, _ = MJN        
    g_xi = blp_moment_joint(params, X, Z, Az, M_iv_est, Xs, As, prices, shares, nus, nus_on_prices, MJN, ownership)
    eta = params[1:1+N_instruments]
    return g_xi - eta

#=============================================================================#
# constraint_s
#=============================================================================#
def constraint_s(params, shares, nus_on_prices, MJN):
    return s(params, nus_on_prices, MJN) - shares.flatten()


# Compute the Jacobian of the constraints using Jax
constraint_g_jac = jacobian(constraint_g)
constraint_s_jac = jacobian(constraint_s)


#=============================================================================#
# objective_jac
#=============================================================================#
def objective_jac(params, W, MJN):
    _, _, N_instruments, _ = MJN
    # Extract etas
    eta = params[1:1+N_instruments]
    gradient_eta = 2 * W @ eta  # Gradient for eta

    # Build the full gradient vector
    gradient = np.zeros_like(params)  # Use NumPy for the full gradient
    gradient[1:1+N_instruments] = np.array(gradient_eta)  # Convert JAX array to NumPy
    return gradient

#=============================================================================#
# objective_hess
#=============================================================================#
def objective_hess(params, W, MJN):
    _, _, N_instruments, _ = MJN
    # Initialize the full Hessian matrix
    hess = np.zeros((len(params), len(params)))
    # Fill in the block corresponding to etas
    hess[1:1+N_instruments, 1:1+N_instruments] = 2 * W  # Constant Hessian for etas
    return hess


#=============================================================================#
# Calculate standard errors
#=============================================================================#
def standard_errors(thetastar, Z, Az, M_iv_est, shares, nus_on_prices, MJN):

    M, J, N_instruments, N = MJN

    # Predicted deltas, xis, and moment conditions
    deltahat = thetastar[1+N_instruments:].reshape(-1, 1)
    xihat = np.array(Az@deltahat)
    g0 = np.array(Z*xihat)

    # Covariance matrix of moment conditions ("meat" of the sandwich formula)
    Bbar = np.cov(g0.T)
      
    # Gradient of shares evaluated at the solution
    grad_s_star = np.array(constraint_s_jac(thetastar, shares, nus_on_prices, MJN))
    
    # Calculate derivative terms
    ds_ddelta = grad_s_star[:, 1+N_instruments:]
    ds_dsigma = grad_s_star[:, 0]
    ddelta_dsigma = -np.linalg.solve(ds_ddelta, ds_dsigma)

    # Constructing the gradient matrix, G
    # This thing represents dg_dtheta, which we sum over j, m to get G. 
    dg0 = np.zeros((J*M, 1+N_instruments+J*M))
    dg0[:, 0] = ddelta_dsigma
    dg0[:, 1+N_instruments:] = np.eye(J*M)
    # Reshape it by product and market
    dg = dg0.reshape(M, J, 1+N_instruments+J*M)
    # Reshape the isnturments
    Z_reshaped = Z.reshape(M, J, N_instruments)
    #Sum over products and markets. 
    G = np.zeros((N_instruments, 1+N_instruments+J*M))
    for i in range(dg.shape[0]):
        for j in range(dg.shape[1]):
            G += np.outer(Z_reshaped[i, j, :], dg[i, j, :])
    G = G/(J*M)
    GTG = G.T @ G

    # Using pseudoinverse because there's a bunch of zero columns and rows, corresponding with eta, which make the matrix non-invertible
    GTG_inv = np.linalg.pinv(GTG)
    
    # Variance-covariance matrix of the GMM estimates
    V_gmm = np.array(GTG_inv @ (G.T) @ Bbar @ G @ GTG_inv)
    
    # Get the parts of the VCV we care about
    v_sigma = V_gmm[0,0]
    V_delta = V_gmm[1+N_instruments:, 1+N_instruments:]
    
    # Get the variance covariance matrix of beta
    V_beta = np.array(M_iv_est @ V_delta @ (M_iv_est.T))
    
    # Get the standard errors
    se_betas = np.sqrt(np.diag(V_beta)/(J*M))
    se_sigma = np.sqrt(v_sigma/(J*M))
    
    return se_sigma, se_betas


#=============================================================================#
# Calculate standard errors, joint estimation
#=============================================================================#
def standard_errors_joint(theta_hat, X, Z, Az, M_iv_est, Xs, prices, shares, nus_on_prices, nus, MJN, ownership):

    M, J, N_instruments, N = MJN
        
    Nd = Z.shape[1]        #Number of demand-side instruments
    Ns = Xs.shape[1] #Number of supply-side instruments
      
    
    # Predicted deltas, xis
    delta_hat = theta_hat[1+N_instruments:].reshape(-1, 1)
    xi_hat = np.array(Az@delta_hat)
    
    ### This means we are a conduct case other than perfect competition
    if ownership.any():
        ### Demand-side moment conditions
        g0_demand = np.array(Z*xi_hat) # Vector of moment conditions, (J*M x 7)
        
        ### Supply-side moments
        As = np.eye(Xs.shape[0]) - Xs@np.linalg.inv(Xs.T@Xs)@Xs.T #supply-side annihilator matrix
        #elas_hat = calculate_price_elasticity(theta_hat, xi_hat, X, M_iv_est, prices, shares, nus, nus_on_prices, MJN) # Elasticities
        mc_hat = calculate_marginal_costs(theta_hat, ownership, xi_hat, X, M_iv_est, prices, nus, nus_on_prices, MJN)                      # Marginal Costs
        
        ### Supply-side moment conditions
        g0_supply = np.array(Xs*mc_hat) # Vector of moment conditions, (J*M x 7)
        
        ### All moments (JM x 10)
        g0 = np.concatenate([g0_demand, g0_supply], axis=1)
    
        # Covariance matrix of moment conditions ("meat" of the sandwich formula)
        Bbar = np.cov(g0.T)
          
        #####-------------- Now, calculating demand-side standard errors
        # Gradient of shares evaluated at the solution
        grad_s_star = np.array(constraint_s_jac(theta_hat, shares, nus_on_prices, MJN))
        
        # Calculate derivative terms
        ds_ddelta = grad_s_star[:, 1+N_instruments:]
        ds_dsigma = grad_s_star[:, 0]
        ddelta_dsigma = -np.linalg.solve(ds_ddelta, ds_dsigma)
    
        # Constructing the gradient matrix, G
        dgd0 = np.zeros((J*M, 1+N_instruments+J*M))
        dgd0[:, 0] = ddelta_dsigma
        dgd0[:, 1+N_instruments:] = np.eye(J*M)
        
        # Reshape it by product and market
        dgd = dgd0.reshape(M, J, 1+N_instruments+J*M)
        Z_reshaped = Z.reshape(M, J, Nd)
            
        #####-------------- Preparing supply-side errors
        #Jacobian of marginal costs evaluated at theta_hat  
        mc_jac = jacobian(calculate_marginal_costs)
        Jmc = mc_jac(theta_hat, ownership, xi_hat, X, M_iv_est, prices, nus, nus_on_prices, MJN).reshape(J*M, J*M+N_instruments+1) 
        # Combined Jacobian of marginal costs with respect to theta
        dmc_dtheta = Jmc
        # This thing (As@dmc_dtheta) gives domega_dtheta, the derivative of the supply-side residual. 
        dgs0 = As@dmc_dtheta
        dgs = dgs0.reshape(M, J, 1+N_instruments+J*M)
        Xs_reshaped = Xs.reshape(M, J, Ns)
            
        # Final "gradient matrix G" used in calculation of standard errors. 
        Gd = np.zeros((Nd, 1+N_instruments+J*M))
        Gs =  np.zeros((Ns, 1+N_instruments+J*M))   
        # Loop through and calculate standard errors    
        for i in range(dgd.shape[0]):
            for j in range(dgd.shape[1]):
                Gd += np.outer(Z_reshaped[i, j, :], dgd[i, j, :])
        for i in range(dgs.shape[0]):
            for j in range(dgs.shape[1]):        
                Gs += np.outer(Xs_reshaped[i, j, :], dgs[i, j, :])
    
        #Combine gradient of supply and demand moment conditions
        G = np.concatenate([Gd, Gs], axis=0)/(J*M)
        GTG = G.T @ G 
    
        # Using pseudoinverse because there's a bunch of zero columns and rows, corresponding with eta, which make the matrix non-invertible
        GTG_inv = np.linalg.pinv(GTG)
        
        # Variance-covariance matrix of the GMM estimates
        V_gmm = np.array(GTG_inv @ (G.T) @ Bbar @ G @ GTG_inv)
        
        # Get the parts of the VCV we care about
        v_sigma = V_gmm[0,0]
        V_delta = V_gmm[1+N_instruments:, 1+N_instruments:]
        
        # Get the variance covariance matrix of beta
        V_beta = np.array(M_iv_est @ V_delta @ (M_iv_est.T))
    
        #Next, use delta method to get standard errors for gamma. 
        Ms = np.linalg.inv(Xs.T@Xs)@Xs.T
        V_mc = (dmc_dtheta)@(V_gmm)@(dmc_dtheta).T
        V_gamma = (Ms)@(V_mc)@(Ms.T)
        
        # Get the standard errors
        se_betas = np.sqrt(np.diag(V_beta)/(J*M))
        se_sigma = np.sqrt(v_sigma/(J*M))
        se_gamma = np.sqrt(np.diag(V_gamma)/(J*M))
        
    else: ### Perfect competition case
        #Demand side moment not dependent on gammas. 
                
        ### Demand-side moment conditions
        g0 = np.array(Z*xi_hat) # Vector of moment conditions, (J*M x 7)
            
        # Covariance matrix of moment conditions ("meat" of the sandwich formula)
        Bbar = np.cov(g0.T)
          
        #####-------------- Now, calculating demand-side standard errors
        # Gradient of shares evaluated at the solution
        grad_s_star = np.array(constraint_s_jac(theta_hat, shares, nus_on_prices, MJN))
        
        # Calculate derivative terms
        ds_ddelta = grad_s_star[:, 1+N_instruments:]
        ds_dsigma = grad_s_star[:, 0]
        ddelta_dsigma = -np.linalg.solve(ds_ddelta, ds_dsigma)
    
        # Constructing the gradient matrix, G
        dgd0 = np.zeros((J*M, 1+N_instruments+J*M))
        dgd0[:, 0] = ddelta_dsigma
        dgd0[:, 1+N_instruments:] = np.eye(J*M)
        
        # Reshape it by product and market
        dgd = dgd0.reshape(M, J, 1+N_instruments+J*M)
        Z_reshaped = Z.reshape(M, J, Nd)
            
        #####-------------- Preparing supply-side errors
        # Final "gradient matrix G" used in calculation of standard errors. 
        Gd = np.zeros((Nd, 1+N_instruments+J*M))
        # Loop through and calculate standard errors    
        for i in range(dgd.shape[0]):
            for j in range(dgd.shape[1]):
                Gd += np.outer(Z_reshaped[i, j, :], dgd[i, j, :])
    
        #Combine gradient of supply and demand moment conditions
        G = Gd/(J*M)
        GTG = G.T @ G 
    
        # Using pseudoinverse because there's a bunch of zero columns and rows, corresponding with eta, which make the matrix non-invertible
        GTG_inv = np.linalg.pinv(GTG)
        
        # Variance-covariance matrix of the GMM estimates
        V_gmm = np.array(GTG_inv @ (G.T) @ Bbar @ G @ GTG_inv)
        
        # Get the parts of the VCV we care about
        v_sigma = V_gmm[0,0]
        V_delta = V_gmm[1+N_instruments:, 1+N_instruments:]
        
        # Get the variance covariance matrix of beta
        V_beta = np.array(M_iv_est @ V_delta @ (M_iv_est.T))
        
        se_betas = np.sqrt(np.diag(V_beta)/(J*M))
        se_sigma = np.sqrt(v_sigma/(J*M))
        
        
        #Supply-side moment standard errors come directly from the SE formulas of linear regression
        #mc=p is no longer a random variable. 
        XsTXs_inv = np.linalg.inv(Xs.T @ Xs)
        #gamma_hat = XTX_inv @ (Xs.T) @ prices
        #residuals
        #omega_hat = prices - Xs @ gamma_hat
        #n, k = Xs.shape
        # Outer product of residuals and rows of X
        #robust_sum = np.zeros((k, k))
        #for i in range(n):
        #    Xi = X[i, :].reshape(-1, 1)  # Row vector of X as column
        #    robust_sum += (omega_hat[i] ** 2) * (Xi @ Xi.T)
        # Robust variance-covariance matrix
        #var_gamma_robust = XTX_inv @ robust_sum @ XTX_inv
        #robust_standard_errors = np.sqrt(np.diag(var_gamma_robust))
        ### Try non-robust SEs
        # Step 1: Calculate regression coefficients (gamma)
        gamma_hat = XsTXs_inv @ Xs.T @ prices
        # Step 2: Calculate residuals
        omega_hat = prices - Xs @ gamma_hat
        # Step 3: Estimate variance of residuals (sigma^2)
        n, k = X.shape
        sigma2 = (omega_hat.T @ omega_hat) / (n - k)
        # Step 4: Calculate variance-covariance matrix of gamma
        var_gamma = sigma2 * np.linalg.inv(Xs.T @ Xs)
        # Step 5: Standard errors of gamma
        se_gamma = np.sqrt(np.diag(var_gamma))
               
    return se_sigma, se_betas, se_gamma


#=============================================================================#
# Calculate price elasticities
#=============================================================================#
##### New version for easier Jacobian calculation in standard errors. 
#@partial(jit, static_argnums=(6,))  --------- non-hashable static arguments are not supported. One of these things is not a Jax array. 
@partial(jit, static_argnums=(8,))
def calculate_price_elasticity(params, xi, X, M_iv_est, prices, shares, nus, nus_on_prices, MJN):

    M, J, N_instruments, N = MJN    

    ### Extract parameters
    sigma = params[0]
    #deltas = params[1+N_instruments:]
    # Use lax.dynamic_slice for dynamic slicing
    deltas_start = N_instruments + 1
    deltas = lax.dynamic_slice(params, (deltas_start,), (params.shape[0] - deltas_start,)).reshape(-1, 1)
    
    ### Calculate betas and alphas
    betas_and_alpha_hat = (M_iv_est @ deltas)
    betas = betas_and_alpha_hat[:3]
    alpha = -betas_and_alpha_hat[3]

    # Take the drawn alphas and calculate the utilities for each consumer
    alphas = (sigma*nus + alpha).reshape(M, N)
    

    # Compute utilities
    utilities = deltas - sigma*nus_on_prices
    # Reshape by markets and products
    utilities_reshaped = utilities.reshape(M, J, N)  # Shape: (M, J, N)
    # Compute the stabilization constant (max utilities per market per individual)
    max_utilities = jnp.max(utilities_reshaped, axis=1, keepdims=True)
    # Stabilized exponentials 
    exp_utilities = jnp.exp(utilities_reshaped-max_utilities)
    #Adjust the outside option (1 becomes exp(-max_utilities))
    outside_option = jnp.exp(-max_utilities)  # Shape: (M, 1, N)
    #Compute the stabilized denominator
    sum_exp_utilities = outside_option + exp_utilities.sum(axis=1, keepdims=True)  # Shape: (M, 1, N)
    # Compute individual-level market shares (before averaging)
    ind_shares = exp_utilities / sum_exp_utilities  # Shape: (M, J, N)    
    ind_shares = ind_shares.reshape(J*M, N)
    
    #### Trying to vectorize
    # Reshaping alphas
    alphas_repeat0 = jnp.repeat(alphas, repeats=J*J, axis=0)    
    alphas_mat = alphas_repeat0.reshape(J, J, M, N, order='F') #(shape:J, J, M, N). 
    # Reshaping prices for j-and k-indexing
    # Here, "j" varies by row, "k" varies by column
    prices_rs = prices.reshape(M, J) 
    prices_repeat = jnp.repeat(prices_rs, repeats=J, axis=1)
    prices_j = prices_repeat.reshape(M, J, J).transpose(1, 2, 0)
    prices_j = prices_j[..., np.newaxis]    #(shape: J, J, M, 1)
    prices_k = prices_j.transpose(1,0,2,3)  #(shape: J, J, M, 1)
    # Do the same for shares
    ind_shares_rs = ind_shares.reshape(M, J, N)
    ind_shares_repeat = jnp.repeat(ind_shares_rs, repeats=J, axis=0)
    ind_shares_k = ind_shares_repeat.reshape(M, J, J, N).transpose(1, 2, 0, 3) #(shape: J, J, M, N)
    ind_shares_j = ind_shares_k.transpose(1,0,2,3)                             #(shape: J, J, M, N)
    # Average of shares
    shares_j = jnp.mean(ind_shares_j, axis=3, keepdims=True)
        
    #### Elasticities
    # Own-price elasticity (we will only need the diagonals of this matrix.)
    elas_own_price = -(prices_j/shares_j)*alphas_mat*ind_shares_j*(1-ind_shares_j) #(shape: J, J, M, N)
    #Cross-price elasticity (we will only need the off-diagonals.)
    elas_cross_price = (prices_k/shares_j)*alphas_mat*ind_shares_j*ind_shares_k    #(shape: J, J, M, N)
    # Average across elasticities (i.e., evaluate the Monte Carlo integral)
    elas_own_price_mean = jnp.mean(elas_own_price, axis=3)     #(shape: J, J, M)
    elas_cross_price_mean = jnp.mean(elas_cross_price, axis=3) #(shape: J, J, M)

    ## Combine the off-diagonal elements of the cross-price elasticities with the diagonal elements of the own-price elasticities. 
    diag_indices = jnp.arange(J)
    diag_mask = (diag_indices[:, None] == diag_indices[None, :])[:, :, None]
    elas_mean = jnp.where(diag_mask, elas_own_price_mean, elas_cross_price_mean)    
    
    return elas_mean.flatten() 

#=============================================================================#
# Calculate marginal costs
#=============================================================================#

def calculate_marginal_costs(params, ownership, xi, X, M_iv_est, prices, nus, nus_on_prices, MJN):
    
    M, J, N_instruments, N = MJN

    # Calculate shares and reshape
    shares = s(params, nus_on_prices, MJN)
    shares_reshaped = shares.reshape(M, J).T

    # Reshape prices
    prices_reshaped = prices.reshape(M, J).T

    # Calculate elasticities and reshape to (J, J, M)
    elasticities = calculate_price_elasticity(params, xi, X, M_iv_est, prices, shares, nus, nus_on_prices, MJN)
    elasticities_reshaped = elasticities.reshape(J, J, M)
        
    mc = jnp.zeros(J*M).reshape(J*M, -1)
    
    for m in range(M):
            elast_mkt = elasticities_reshaped[:, :, m].reshape(J, J)
            shares_mkt = shares_reshaped[:, m]
            prices_mkt = prices_reshaped[:, m]
            p_over_s = (1/shares_mkt).reshape(J, -1) @ prices_mkt.reshape(-1, J)
            s_partial_p = elast_mkt/p_over_s
            mc_mkt = jnp.linalg.inv(ownership*s_partial_p.T) @ shares_mkt.reshape(J, -1) + prices_mkt.reshape(J, -1)
            mc = mc.at[J*m:J*m + J].add(mc_mkt)

    return mc

#=============================================================================#
# Calculate consumer surplus
#=============================================================================#

def calculate_consumer_surplus(betas, alpha, sigma_alpha, xi, X, prices, MJN):

    M, J, N_instruments, N = MJN  
    
    # Draw alphas and calculate the utilities for each consumer
    alphas = (sigma_alpha*np.random.lognormal(0.0, 1.0, M*N) + alpha).reshape(M, N)
    
    utilities = (betas.reshape(1, 3) @ X.T).reshape(J*M, -1) - prices*np.repeat(alphas, repeats=J, axis=0) + xi
    utilities_exp = np.exp(utilities)

    # Create an array of shape (M, N) that will store consumer surplus for each consumer in all markets
    cs = np.zeros((M, N))
    
    for m in range(M):
        utilities_exp_mkt = utilities_exp[J*m:J*m + J, :]
        cs[m, :] = np.log(1 + utilities_exp_mkt.sum(axis=0))/alphas[m, :]

    cs = cs.sum(axis=1)/N
    return cs



#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#
# Functions for the first part: making graphs, loading the data, etc. 
#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#

#=============================================================================#
# Loading data from MATLAB
#=============================================================================#
def load_mat_data(datapath, nrProducts, nrMarkets):
    """
    Purpose: Loads a .mat data file and returns a Pandas DataFrame.

    Parameters
    ----------
    datapath : str
        The path to the .mat file.

    nrProducts : int
        The number of products in the market data.
    
    nrMarkets : int
        The number of markets in the market data.
        
    Returns
    -------
    pd.DataFrame
        The market level data in the .mat data file converted to a Pandas DataFrame.
    pd.DataFrame
        The simulated alphas in the .mat data file converted to a Pandas DataFrame.
        
    Description
    -----------
    This function loads the .mat data using scipy.io.loadmat and collects the variable names and the 
    data (in numpy arrays), ignoring other items in the dictionary (such as the header). It then converts
    the cleaned dictionary into two DataFrames, one for the market level data and one for the simulated alphas in each market.
    """

    # Load the .mat data and format the X's appropriately
    mat = loadmat(datapath)
    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    mat['x2'] = mat['x1'][:, 1]
    mat['x3'] = mat['x1'][:, 2]
    mat['x1'] = mat['x1'][:, 0]

    # Get the simulated alphas into one DataFrame
    alphas = mat['alphas']
    column_names = [i for i in range(alphas.shape[1])]
    df_alphas = pd.DataFrame(alphas, columns=column_names)
    mat.pop('alphas')

    # Store the market level data to a DataFrame
    df_mkt = pd.DataFrame({k: np.array(v).flatten(order='F') for k, v in mat.items()})

    # Add market and product ids to the market level data
    product_ids = [i+1 for i in range(nrProducts)] * nrMarkets
    market_ids = [i+1 for i in range(nrMarkets) for _ in range(nrProducts)]
    df_mkt['market_id'] = market_ids
    df_mkt['product_id'] = product_ids
    
    return df_mkt, df_alphas

#=============================================================================#
# Drawing logit shocks (epsilons)
#=============================================================================#
def draw_epsilons(alphas):
    """
    Purpose: Draws epsilons from the specified distribution, to the same shape as alphas.

    Parameters
    ----------
    alphas : pd.DataFrame
        The DataFrame of alphas, in the form (Number of Consumers) x (Number of Markets).
        
    Returns
    -------
    pd.DataFrame
        The simulated epsilons in a Pandas DataFrame.
        
    Description
    -----------
    This function takes the simulated alphas for all consumers in all markets and simulates epsilons for each of them.
    """
    draws = np.random.gumbel(size=(alphas.shape[0], alphas.shape[1]))
    column_names = [i+1 for i in range(draws.shape[1])]
    df_epsilons = pd.DataFrame(draws, columns=column_names)
    return df_epsilons

#=============================================================================#
# calculate consumer welfare for the true data. 
#=============================================================================#
def calculate_welfare(data, alphas, beta, epsilons):
    """
    Purpose: Calculates the consumer welfares based on the given market level data and the simulated consumers.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame of market level data.

    alphas : pd.DataFrame
        The DataFrame of alphas, in the form (Number of Consumers) x (Number of Markets).

    epsilons : pd.DataFrame
        The DataFrame of epsilons, in the form (Number of Consumers) x (Number of Markets).
        
    Returns
    -------
    np.array
        The (Number of Consumers) x (Number of Markets) array of utilities (welfare) of each consumer in each market, conditional
        on them choosing optimally based on their utility function.
        
    Description
    -----------
    This function takes the market level data, and simulated alphas and epsilons and calculates the welfare for each consumer in each market,
    conditional on them choosing optimally based on their utility function parameters.
    """
    # Check that the alphas and epsilons agree on the number of markets
    assert len(alphas) == len(epsilons)

    # Store number of markets
    nrMarkets = len(alphas)

    # Store number of products
    nrProducts = data['product_id'].max()
    
    for market_id in range(1, nrMarkets+1):
        # Get the data for the market at hand
        mkt_data = data.loc[data['market_id']==market_id].copy()
        
        # Calculate the part of the utility that is independent from the consumer
        mkt_data['common_util'] = beta[0]*mkt_data['x1'] + beta[1]*mkt_data['x2'] + beta[2]*mkt_data['x3'] + mkt_data['xi_all']
        
        # Calculate consumer utilities
        utils = {}
        for product in range(1, nrProducts+1):
            utils[product] = (
                -alphas.iloc[market_id-1].values*mkt_data.loc[mkt_data['product_id']==product]['P_opt'].iloc[0]
                + epsilons.iloc[market_id-1].values
                + mkt_data.loc[mkt_data['product_id']==product]['common_util'].iloc[0]
            )

        # Stack utilities for each product in the market for each consumers into a matrix
        product_utilities = np.stack(tuple(utils.values()), axis=1)

        # Create a column of zeros with the same number of rows as there are consumers
        zero_column = np.zeros((product_utilities.shape[0], 1))
        
        # Concatenate the zero column to 'product_utilities', to store the utility from the outside option (zero)
        product_utilities = np.concatenate([product_utilities, zero_column], axis=1)

        mkt_welfare = np.amax(product_utilities, axis=1).reshape(500, -1)

        # Divide by alpha_i's to get the surplus in monetary units
        mkt_welfare = mkt_welfare/alphas.iloc[market_id-1].values
        
        if market_id == 1:
            market_welfares = mkt_welfare
        else:
            market_welfares = np.hstack([market_welfares, mkt_welfare])
            
    return market_welfares

#=============================================================================#
# Plot the histograms for the true data. 
#=============================================================================#
def plot_two_histograms(data1, data2, bins=500, labels=('Data 1', 'Data 2')):
    """
    Purpose: Plots two histograms side by side to compare the distributions of two different datasets.

    Parameters
    ----------
    data1 : np.array
        The first dataset, which can be a multi-dimensional NumPy array. All elements will be flattened for the histogram.
    
    data2 : np.array
        The second dataset, which can also be a multi-dimensional NumPy array. All elements will be flattened for the histogram.
    
    bins : int, optional
        The number of bins for each histogram (default is 50).
    
    labels : tuple of str, optional
        Labels for each dataset, used in the titles of the histograms (default is ('Data 1', 'Data 2')).
        
    Description
    -----------
    This function takes two datasets, flattens them into 1-dimensional arrays if necessary, and plots them as two
    histograms side by side in a single figure. It provides a visual comparison of the distributions in both datasets.
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the first histogram
    axes[0].hist(data1.flatten(), bins=bins)
    axes[0].set_title(f'{labels[0]}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    
    # Plot the second histogram
    axes[1].hist(data2.flatten(), bins=bins)
    axes[1].set_title(f'{labels[1]}')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    
    # Show the plots
    plt.tight_layout()
    plt.show()

def predict_prices_and_shares(ownership, mc, betas, alpha, sigma_alpha, xi, X, MJN):
    
    M, J, N_instruments, N = MJN

    # Draw alphas and calculate the utilities for each consumer
    alphas = (sigma_alpha*np.random.lognormal(0.0, 1.0, M*N) + alpha).reshape(M, N)

    def predict_ind_shares(p):
        utilities = (betas.reshape(1, 3) @ X.T).reshape(J*M, -1) - p.reshape(-1, 1)*np.repeat(alphas, repeats=J, axis=0) + xi

        # Reshape utilities for markets and products
        utilities_reshaped = utilities.reshape(M, J, N)  # Shape: (M, J, N)
    
        # Compute the stabilization constant (max utility per market per individual)
        max_utilities = jnp.max(utilities_reshaped, axis=1, keepdims=True)  # Shape: (M, 1, N)
    
        # Stabilized exponentials
        exp_utilities = jnp.exp(utilities_reshaped - max_utilities)  # Shape: (M, J, N)
    
        # Adjust the "outside option" (1 becomes exp(-max_utilities))
        outside_option = jnp.exp(-max_utilities)  # Shape: (M, 1, N)
    
        # Compute the stabilized denominator
        sum_exp_utilities = outside_option + exp_utilities.sum(axis=1, keepdims=True)  # Shape: (M, 1, N)

        # Compute shares
        ind_shares = exp_utilities / sum_exp_utilities  # Shape: (M, J, N)

        return ind_shares.reshape(J*M, N)
    
    def zeta(p):
        
        ind_shares = predict_ind_shares(p)
        mkt_shares = ind_shares.mean(axis=1)

        lambda_diag = (ind_shares*np.repeat(-alphas, repeats=J, axis=0)).sum(axis=1)/N
        
        # Reshape into a (J*M, J) matrix with block diagonal structure
        blocks = []
        blocks_inv = []
        for i in range(0, len(lambda_diag), J):
            diag_matrix = np.diag(lambda_diag[i:i+J])
            diag_matrix_inv = np.linalg.inv(diag_matrix)
            blocks.append(diag_matrix)
            blocks_inv.append(diag_matrix_inv)
        
        # Stack the blocks vertically
        lambda_mat = np.vstack(blocks)
        lambda_mat_inv = np.vstack(blocks_inv)

        gamma_mat = np.zeros((M, J, J))
        for m in range(M):
            for j in range(J):
                for k in range(J):
                    gamma_mat[m, j, k] = (ind_shares[J*m + j, :]*ind_shares[J*m + k, :]*(-alphas[m, :])).mean()

        gamma_mat = gamma_mat.reshape(J*M, J)
        
        zeta = np.zeros(J*M)
        for m in range(M):
            zeta[m*J:m*J + J] = (lambda_mat_inv[m*J:m*J + J, :] @ (ownership*gamma_mat[m*J:m*J + J, :]) @ (p[m*J:m*J + J].reshape(-1, 1) - mc[m*J:m*J + J]) 
                                - lambda_mat_inv[m*J:m*J + J, :] @ mkt_shares[m*J:m*J + J].reshape(-1, 1)).reshape(-1)

        return zeta, lambda_mat

    def contraction_mapping(zeta, p0, tol=1e-8, max_iter=1000):
        """
        Implements the contraction mapping: p <- c + zeta(p).
    
        Parameters:
        - zeta: function zeta(p), mapping p to a vector or scalar of the same shape as p.
        - p0: initial guess for p (scalar or array).
        - tol: convergence tolerance (default 1e-8).
        - max_iter: maximum number of iterations (default 1000).
    
        Returns:
        - p: converged value of p.
        """
        p = p0
        for n_iter in range(max_iter):
            zeta_vec, lambda_mat = zeta(p)
            p_new = mc.reshape(-1) + zeta_vec

            # Reshape into groups of 3x3 matrices and 3x1 vectors
            lambda_mat_grouped = lambda_mat.reshape(-1, J, J)  # Shape: (100, 3, 3)
            p_grouped = (p_new - p).reshape(-1, 3, 1)  # Shape: (100, 3, 1)
            
            # Perform batch matrix-vector multiplication
            norm_grouped = np.matmul(lambda_mat_grouped, p_grouped)  # Shape: (100, 3, 1)
            
            # Flatten the result back into a vector
            norm_vector = norm_grouped.reshape(-1)  # Shape: (300,)
            norm = np.linalg.norm(norm_vector, np.inf)
            print(f"Iteration {n_iter}. Norm: {norm}.")
            if norm < tol:  # Check for convergence
                print("Contraction mapping converged, found prices that satisfy the FOC.")
                print("Iterations:", n_iter)
                return p_new
            p = p_new
        raise RuntimeError("Contraction mapping did not converge within the maximum number of iterations.")

    p_init = np.ones(J*M)
    
    res_prices = contraction_mapping(zeta, p_init)
    res_shares = predict_ind_shares(res_prices).sum(axis=1)/N
    
    return res_prices, res_shares
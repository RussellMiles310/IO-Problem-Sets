# -*- coding: utf-8 -*-
"""
Source functions
"""
import pandas as pd
import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
from jax import grad, jacobian, hessian, config, jit, lax
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
def blp_instruments_all(X, W_costs, prices, MJN):
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

    # Final element: Prices
    prices_reshaped = prices.reshape(M, J, -1)
    
    # Concatenate results along the last dimension and add a column of ones
    instruments = jnp.concatenate([
        jnp.ones((M, J, 1)),
        X_jm,
        X_j_sum,
        X_m_sum,
        W_costs_reshaped,
        prices_reshaped
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
    dG0 = np.zeros((J*M, 1+N_instruments+J*M))
    dG0[:, 0] = ddelta_dsigma
    dG0[:, 1+N_instruments:] = np.eye(J*M)
    
    # Reshape it by product and market
    dg = dG0.reshape(M, J, 1+N_instruments+J*M)

    Z_reshaped = Z.reshape(M, J, N_instruments)

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
# Calculate price elasticities
#=============================================================================#
def calculate_price_elasticity(betas, alpha, sigma_alpha, xi, X, prices, shares, MJN):

    M, J, N_instruments, N = MJN    

    # Draw alphas and calculate the utilities for each consumer
    alphas = (sigma_alpha*np.random.lognormal(0.0, 1.0, M*N) + alpha).reshape(M, N)
    
    utilities = (betas.reshape(1, 3) @ X.T).reshape(J*M, -1) - prices*np.repeat(alphas, repeats=J, axis=0) + xi

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
    ind_shares = ind_shares.reshape(J*M, N)
    
    # Create a (J*M) x (J*M) matrix that will store the elasticities
    elasticities = np.zeros((J, J, M))
    
    # Calculate price elasticities
    for m in range(M):
        for j in range(J):
            for k in range(J):
                if j == k:
                    elast = (-prices[j]/shares[j])*alphas[m]*ind_shares[j, :]*(1 - ind_shares[j, :])
                else:
                    elast = (prices[k]/shares[j])*alphas[m]*ind_shares[j, :]*ind_shares[k, :]
                elasticities[j, k, m] = elast.sum()/N
                
    return elasticities


#=============================================================================#
# Calculate marginal costs
#=============================================================================#

def calculate_marginal_costs(elasticities, conduct, prices, shares, MJN):
    
    M, J, N_instruments, N = MJN    

    if conduct == "perfect":
        return prices
    elif conduct == "collusion":
        ownership = np.ones((J, J))
    elif conduct == "oligopoly":
        ownership = np.eye(J)
    else:
        print("The specified conduct is not an option ('perfect', 'collusion', 'oligopoly').")
        print("Returning the vector of prices (i.e., the marginal costs for the perfect competition case).")
        
    mc = np.zeros(J*M).reshape(J*M, -1)
    
    for m in range(M):
            elast_mkt = elasticities[:, :, m].reshape(J, J)
            mc_mkt = np.linalg.inv(ownership*elast_mkt) @ shares[J*m:J*m + J].reshape(J, -1) + prices[J*m:J*m + J].reshape(J, -1)
            mc[J*m:J*m + J] = mc_mkt

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



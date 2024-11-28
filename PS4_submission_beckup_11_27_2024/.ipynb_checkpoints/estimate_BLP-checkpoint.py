# -*- coding: utf-8 -*-
"""
Function to estimate BLP
"""

from source_functions import *

def estimate_BLP(df, alphas, sigma_alpha_init, mode, verbose_print = 1, scale_delta_guess = 1, delta_solve_init = False, max_iter = 10000, beta_true = (5, 1, 1), gamma_true = (2, 1, 1), alpha_true=1, sigma_alpha_true=1):

    ###
    J=3
    M = alphas.shape[0] #Number of markets
    N = alphas.shape[1] #Number of consumers

    if mode == "demand_side":
        N_instruments = 7
    elif mode == "p_exercise":
        N_instruments = 4
    elif mode == "supply_W":
        N_instruments = 8
    elif mode == "supply_joint":
        N_instruments = 8


    #Initial guesses for eta and delta
    eta_init =  jnp.ones((N_instruments, 1))
    delta_init = jnp.ones((J*M, 1))  * scale_delta_guess ### For trying different initial guesses of delta to test stability

    #Initial guess for all parameters    
    params_init = jnp.concatenate([np.array([sigma_alpha_init]).reshape(-1,1), eta_init, delta_init], axis=0).flatten()

    MJN = (M, J, N_instruments, N) #Stack all shape-related parameters together

    #========== Constructing correct values of parameters, for use later ============================================================================#
    deltas_correct = (5 + df['x2'] + df['x3'] - df['P_opt'] + df['xi_all']).values.reshape(J*M, -1)
    # Prepend the new element
    first_elems = jnp.ones(1+N_instruments).reshape(-1, 1)
    params_correct = jnp.vstack([first_elems, deltas_correct]).reshape(-1)

    xi_true = (df['xi_all']).values.reshape(J*M, -1)

    #========== Constructing commonly used variables ===============================================================================================#
    X = jnp.array(df[['x1', 'x2', 'x3']].values) #Matrix of product characteristics
    prices = jnp.array(df[['P_opt']].values)
    shares = jnp.array(df[['shares']].values)
    W_costs = jnp.array(df[['w']].values)


    # For speed, compute this outside of the function and pass it later
    alphas_repeat = jnp.repeat(np.array(alphas.values), repeats=J, axis=0)

    # Random coefficients nu on the prices
    nus_on_prices = (alphas_repeat-1) * prices.reshape(-1, 1)

    # Get matrix of regressors
    Xbar = df[['x1', 'x2', 'x3', 'P_opt']].values.reshape(J*M, 4)

    # Get the matrix of instruments (including x1, x2, and the BLP moments). 
    Z_everything = jnp.array(blp_instruments_all(X, W_costs, prices, MJN))

    if mode == "demand_side":
        Z = Z_everything[:, 0:7]
    elif mode == "p_exercise":
        Z = Z_everything[:, [0, 1, 2, 9]]
    elif mode == "supply_W":
        Z = Z_everything[:, 0:8]
    elif mode == "supply_joint":
        Z = Z_everything[:, 0:8]


    #Projection matrix onto the instruments
    Pz = Z @ jnp.linalg.inv(Z.T @ Z) @ Z.T
    #Annihiliator matrix to get xi from delta
    Az = jnp.eye(Pz.shape[0]) - Xbar @ jnp.linalg.inv(Xbar.T @ Pz @ Xbar) @ Xbar.T @ Pz
    #This thing gets the value of beta.
    M_iv_est = (np.linalg.inv(Xbar.T @ Pz @ Xbar) @ Xbar.T @ Pz)
    # GMM weighting matrix
    W = np.eye(N_instruments)


    if delta_solve_init:
        #### Solve initial deltas for improved initial guess. 
        print("#============================================================================#")
        print("#== Getting feasible initial guess using modified MPEC")
        print("#============================================================================#")
        res = solve_init_deltas(params_init, shares, nus_on_prices, MJN)
    
        ### Replace the deltas with the new ones
        params_init0 = params_init
        params_init = res.x
        #params_init[1+N_instruments:] = res.x 


    #========== Define the constraints for the MPEC ===============================================================================================#
    constraints = [
        {
            'type': 'eq',  # Equality constraint g(x) = eta
            'fun': lambda x: np.asarray(constraint_g(x, Z, Az, MJN)),  # Convert to NumPy
            'jac': lambda x: np.asarray(constraint_g_jac(x, Z, Az, MJN))  # Convert Jacobian to NumPy
        },
        {
            'type': 'eq',  # Equality constraint s(x) = shares
            'fun': lambda x: np.asarray(constraint_s(x, shares, nus_on_prices, MJN)),
            'jac': lambda x: np.asarray(constraint_s_jac(x, shares, nus_on_prices, MJN))
        }
    ]
    if mode == "supply_joint":
        print("Supply-side version: Adding additional constraint to MPEC")
        #constraints.append(
            #New constraint needed for the supply side, joint MPEC
        #)


    #========== Running the MPEC optimization routine ==============================================================================================#

    print("#============================================================================#")
    print("#== Solving the MPEC optimization routine")
    print("#============================================================================#")


    # Silence warning about delta_grad==0.0
    warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

    # Solve optimization
    tolerance = 1e-40
    W = np.eye(N_instruments)

    result = minimize(
        fun = lambda x: objective_mpec(x, W, MJN),
        x0 = params_init,
        constraints = constraints,
        method = 'trust-constr',
        jac = lambda x: objective_jac(x, W, MJN),
        hess = lambda x: objective_hess(x, W, MJN),
        options = {
                 'xtol': tolerance,
                 'gtol': tolerance,
                 'barrier_tol': tolerance,
                 'sparse_jacobian': True,
                 'disp': True,
                 'verbose': verbose_print, 
                 'maxiter':max_iter
                }, 
    )

    # Output results
    if result.success:
        print("Optimal solution found.")
    else:
        print("Optimization failed:", result.message)


    #========== Calculate the parameters of interest ==============================================================================================#
    theta_hat = result.x # Get the estimated value of theta
    delta_hat = theta_hat[N_instruments + 1:].reshape(J*M, 1)
    beta_and_alpha_hat = np.array(M_iv_est @ delta_hat)
    xi_hat = np.array(Az@delta_hat)

    beta_hat = beta_and_alpha_hat[:3]
    alpha_hat = -beta_and_alpha_hat[3].item()
    sigma_alpha_hat = theta_hat[0].item()


    print("#============================================================================#")
    print("#== Optimal parameters found. Next, calculating standard errors:")
    print("#== Calculating standard errors, elasticities, profits, and consumer surplus")
    print("#============================================================================#")

    #========== Calculate the standard errors =====================================================================================================#
    se_sigma, se_betas = standard_errors(theta_hat, Z, Az, M_iv_est, shares, nus_on_prices, MJN)
    se = np.append([se_sigma], se_betas)

    #========== Calculate the elasticities =====================================================================================================#
    #Predicted deltas, xis, and moment conditions
    beta_true_array = np.array(beta_true)
    #True value of the elasticities
    elasticities_true = calculate_price_elasticity(beta_true_array, alpha_true, sigma_alpha_true, xi_true, X, prices, shares, MJN)
    #Mean of the true value of elasticities
    mean_elasticities_true = elasticities_true.mean(axis=2)
    #######
    #Predicted value of the elasticities
    elasticities_hat = calculate_price_elasticity(beta_hat, alpha_hat, sigma_alpha_hat, xi_hat, X, prices, shares, MJN)
    #Mean of the true value of elasticities
    mean_elasticities_hat = elasticities_hat.mean(axis=2)

    #========== Calculate the marginal costs =====================================================================================================#
    #oligopoly
    mc_true_olig = calculate_marginal_costs(elasticities_true, "oligopoly", prices, shares, MJN)
    mc_hat_olig = calculate_marginal_costs(elasticities_hat, "oligopoly", prices, shares, MJN)
    #Perfect competition
    mc_true_pc = calculate_marginal_costs(elasticities_true, "perfect", prices, shares, MJN)
    mc_hat_pc = calculate_marginal_costs(elasticities_hat, "perfect", prices, shares, MJN)
    #collusion
    #Perfect competition
    mc_true_co = calculate_marginal_costs(elasticities_true, "perfect", prices, shares, MJN)
    mc_hat_co = calculate_marginal_costs(elasticities_hat, "perfect", prices, shares, MJN)
    
    #========== Calculate the consumer surplus ===================================================================================================#
    cs_true = calculate_consumer_surplus(beta_true_array, alpha_true, sigma_alpha_true, xi_true, X, prices, MJN)
    cs_hat = calculate_consumer_surplus(beta_hat, alpha_hat, sigma_alpha_hat, xi_hat, X, prices, MJN)



    #========== Create a dictionary to return all output ==========================================================================================#
    OUT = {
        'delta_hat': delta_hat,
        'beta_hat': beta_hat,
        'alpha_hat': alpha_hat,
        'sigma_alpha_hat': sigma_alpha_hat,
        'se': se,
        'elasticities': {'true': elasticities_true, 'hat': elasticities_hat}, 
        'mean_elasticities': {'true': mean_elasticities_true, 'hat': mean_elasticities_hat}, 
        'mc': {
            "perfect":{'true': mc_true_pc, 'hat': mc_hat_pc},
            "oligopoly":{'true': mc_true_olig, 'hat': mc_hat_olig},
            "collusion":{'true': mc_true_co, 'hat': mc_hat_co}           
            }, 
        'cs': {'true': cs_true, 'hat': cs_hat}, 
        'optim_results': result
    }

    print("#============================================================================#")
    print("#== BLP Estimation Complete")
    print("#============================================================================#")


    return OUT
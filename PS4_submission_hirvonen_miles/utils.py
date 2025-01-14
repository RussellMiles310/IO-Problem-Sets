import numpy as np
import pandas as pd

def store_results(results, file_path):

    table = pd.DataFrame(columns=['Parameter', 'Estimate', 'Std. Error', 'True Value', 'Bias'])
    
    sigma_alpha_true=1
    beta_true = np.array([5, 1, 1])
    alpha_true=1
    gamma_true = np.array([2, 1, 1])
    
    for i, name in enumerate(results['se_names']):

        if i == 0:
            estimate = results['sigma_alpha_hat']
            true = sigma_alpha_true
        elif i >= 1 and i <= 3:
            estimate = (results['beta_hat'][i-1]).item()
            true = beta_true[i-1].item()
        elif i == 4:
            estimate = results['alpha_hat']
            true = alpha_true
        elif i >= 5 and i <= 7:
            estimate = (results['gamma_hat'][i-5]).item()
            true = gamma_true[i-5].item()
        
        bias = estimate-true
        #print(estimate)
        
        se = results['se'][i]

        table.loc[len(table)] = [name, estimate, se, true, bias]

        file_type = file_path.split(".")[-1]

        if file_type == "xlsx":        
            table.to_excel(file_path, index=False)
            print(f"Excel file saved to {file_path}")
        elif file_type == "tex":
            # Generate LaTeX code for the table
            latex_code = table.to_latex(index=False, caption="Parameter Estimates", float_format="%.4f")
            # Save the LaTeX code to a file
            with open(file_path, "w") as file:
                file.write(latex_code)
            print(f"TeX file saved to {file_path}")
        else:
            print(f"File type {file_type} is not supported. Options are: .xlsx and .tex.")
            
def save_mc_to_excel(results, file_name="mc_comparison.xlsx", mode = "mc"):
    """
    Extracts true and predicted marginal costs from the dictionary 'mc',
    creates a comparison DataFrame, and saves it to an Excel file.

    Parameters:
    mc (dict): A dictionary with keys 'true' and 'hat', each containing a 3x1 vector.
    file_name (str): Name of the Excel file to save. Default is 'mc_comparison.xlsx'.

    Returns:
    None
    """
    if mode == "mc":
        mc = results['mean_mc']
    else:
        mc = results['mean_profits']
    
    # Extract true and predicted values from the dictionary
    true_values = mc.get('true', [])
    predicted_values = mc.get('hat', [])
    
    # Ensure they are the same length
    if len(true_values) != len(predicted_values):
        raise ValueError("Length of 'true' and 'hat' values in 'mc' dictionary must be the same.")

    # Create a DataFrame for comparison
    df = pd.DataFrame({
        "Index": range(1, len(true_values) + 1),
        "True Value": true_values.flatten(),
        "Predicted Value": predicted_values.flatten(),
        "Difference": (predicted_values - true_values).flatten()
    })
    
    # Save the DataFrame to an Excel file
    df.to_excel(file_name, index=False, sheet_name="MC Comparison")
    print(f"Comparison file saved to {file_name}.")
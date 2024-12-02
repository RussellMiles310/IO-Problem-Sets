import pandas as pd

def store_results(results, file_path):

    table = pd.DataFrame(columns=['Parameter', 'Estimate', 'Std. Error'])
    
    for i, name in enumerate(results['se_names']):

        if i == 0:
            estimate = results['sigma_alpha_hat']
        elif i <= 1 and i <= 3:
            estimate = results['beta_hat'][i-1]
        elif i == 4:
            estimate = results['alpha_hat']
        elif i <= 5 and i <= 7:
            estimate = results['gamma_hat'][i-5]

        se = results['se'][i]

        table.loc[len(table)] = [name, estimate, se]

        file_type = file_path.split(".")[-1]

        if file_type == "xlsx":        
            df.to_excel(file_path, index=False)
            print(f"Excel file saved to {file_path}")
        elif file_type == "tex":
            # Generate LaTeX code for the table
            latex_code = df.to_latex(index=False, caption="Parameter Estimates", float_format="%.2f")
            # Save the LaTeX code to a file
            with open(file_path, "w") as file:
                file.write(latex_code)
            print(f"TeX file saved to {file_path}")
        else:
            print(f"File type {file_type} is not supported. Options are: .xlsx and .tex."
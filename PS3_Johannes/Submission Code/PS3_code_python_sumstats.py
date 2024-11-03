#!/usr/bin/env python
# coding: utf-8

# # Problem Set 3

# ### Preliminaries

# Load the needed libraries and the data.

import pandas as pd
import numpy as np
import re

df = pd.read_stata('PS3_data.dta')


# Rename variables

# Use the given codebook to rename variables.

labels = {
    "X03": "Output",
    "X04": "Industry_1_dummy",
    "X05": "Industry_2_dummy",
    "X06": "Industry_3_dummy",
    "X07": "Industry_4_dummy",
    "X08": "Industry_5_dummy",
    "X09": "Industry_6_dummy",
    "X10": "Industry_7_dummy",
    "X11": "Industry_8_dummy",
    "X12": "Industry_9_dummy",
    "X13": "Industry_10_dummy",
    "X14": "Industry_11_dummy",
    "X15": "Industry_12_dummy",
    "X16": "Industry_13_dummy",
    "X17": "Industry_14_dummy",
    "X18": "Industry_15_dummy",
    "X19": "Industry_16_dummy",
    "X20": "Industry_17_dummy",
    "X21": "Industry_18_dummy",
    "X22": "200_workers_and_fewer",
    "X23": "More_than_200_workers",
    "X24": "Year_1990_dummy",
    "X25": "Year_1991_dummy",
    "X26": "Year_1992_dummy",
    "X27": "Year_1993_dummy",
    "X28": "Year_1994_dummy",
    "X29": "Year_1995_dummy",
    "X30": "Year_1996_dummy",
    "X31": "Year_1997_dummy",
    "X32": "Year_1998_dummy",
    "X33": "Year_1999_dummy",
    "X34": "Merger_dummy",
    "X35": "Scission_dummy",
    "X36": "RD_expenditure",
    "X37": "Process_innovation_dummy",
    "X38": "Product_innovation_dummy",
    "X39": "Investment",
    "X40": "Capital",
    "X41": "Number_of_workers",
    "X42": "Effective_hours_per_worker",
    "X43": "Total_effective_hours",
    "X44": "Intermediate_consumption",
    "X45": "Output_price_index",
    "X46": "Consumer_price_index",
    "X47": "Region_of_industrial_employment",
    "X48": "Hourly_wage",
    "X49": "Materials_price_index",
    "X50": "Proportion_of_temporary_workers",
    "X51": "Proportion_of_white_collar_workers",
    "X52": "Proportion_of_engineers_and_graduates",
    "X53": "Proportion_of_non_graduates",
    "X54": "Technological_sophistication",
    "X55": "Market_dynamism_index",
    "X56": "Incorporated",
    "X57": "Ownership_control_identification",
    "X58": "Firm_age",
    "X59": "NACE_code",
    "X60": "Entrant_firm_dummy",
    "X61": "Exiting_firm_dummy"
}

df.rename(columns=labels, inplace=True)


# Sample Statistics

ind_dummies = [c for c in df.columns if c.startswith('Industry_')]
years = [y for y in range(1990, 2000)]

def print_sample_stats(varlist, balanced=False):
    if balanced:
        data = df[df['obs'] == 10].copy()
    else:
        data = df.copy()
    for var in varlist:
        print('***************************')
        print(var.replace('_', ' '))
        print('***************************\n')
        print(data[var].describe())
        print()

# Industry-year firm and non-zero variable counts

def print_industry_year_stats(balanced=False):
    if balanced:
        data = df[df['obs'] == 10].copy()
    else:
        data = df.copy()
    for ind in ind_dummies:
        print(f'Industry: {re.sub("[^0-9]", "", ind)}')
        for year in years:
            print(
                f"Year {str(year)}: \
                No. of observations: {str(len(data[(data[ind] == 1) & (data['year'] == year)]))}, \
                Non-zero investment: {str(len(data[(data[ind] == 1) & (data['year'] == year) & (data['Investment'] > 0)]))}, \
                Non-zero hours: {str(len(data[(data[ind] == 1) & (data['year'] == year) & (data['Total_effective_hours'] > 0)]))}, \
                Non-zero materials: {str(len(data[(data[ind] == 1) & (data['year'] == year) & (data['Intermediate_consumption'] > 0)]))}"
            )

# Variables of interest
variables = ['Output', 'Investment', 'Capital', 'Total_effective_hours', 'Intermediate_consumption']


#### Unbalanced panel

print_sample_stats(variables)

print_industry_year_stats()


### Balanced panel

print_sample_stats(variables, balanced=True)

print_industry_year_stats(balanced=True)


# NOTE: Seems like firms are switching industries, which causes the panel not to be balanced within industries!

# Export the data to .dta for Stata parts of the code

df.to_stata('PS3_data_clean.dta', write_index=False)
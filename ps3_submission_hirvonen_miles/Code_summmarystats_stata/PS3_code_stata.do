* Set path
cd "../PS3/"

* Load the cleaned data (with variable names)
clear all
use "PS3_data_clean.dta"

	* Pick industry 13
	keep if Industry_13_dummy == 1

	* Rename the variables to shorter versions
	rename Output Y
	rename Capital K
	rename Total_effective_hours L
	rename Intermediate_consumption M
	rename Investment I
	
* NOTE: The variables are already logged, so we can just proceed with estimation
	
* BALANCED
	
	preserve
	
	* Since there's industry switching, ensure that the panel is balanced within the industry
	bysort firm_id: keep if _N == 10

	* OLS

	reg Y K L M, vce(robust)

	* Fixed Effects

	reg Y K L M i.firm_id i.year, vce(robust)

	* First Differences

	xtset firm_id year
	
	reg d.Y d.K d.L d.M, vce(robust)

	* Long Differences

	foreach var in Y K L M {
		gen `var'_ld = `var' - L5.`var'
	}

	reg Y_ld K_ld L_ld M_ld, vce(robust)

	* Random Effects

	xtreg Y K L M i.year, re robust
	
	* Hausman Test
	qui xtreg Y K L M i.year, fe
	estimates store fixed_effects
	qui xtreg Y K L M i.year, re
	hausman fixed_effects ., sigmamore
	
	restore
	
* UNBALANCED

	* OLS

	reg Y K L M, vce(robust)

	* Fixed Effects

	reg Y K L M i.firm_id i.year, vce(robust)

	* First Differences
	
	xtset firm_id year
	
	reg d.Y d.K d.L d.M, vce(robust)

	* Long Differences

	foreach var in Y K L M {
		gen `var'_ld = `var' - L5.`var'
	}

	reg Y_ld K_ld L_ld M_ld, vce(robust)

	* Random Effects

	xtreg Y K L M i.year, re robust

	* Hausman Test
	qui xtreg Y K L M i.year, fe
	estimates store fixed_effects
	qui xtreg Y K L M i.year, re
	hausman fixed_effects ., sigmamore
	
* ARELLANO AND BOND (1991)

	* Create value added output
	gen Y_va = ln(exp(Y)*exp(Output_price_index) - exp(M)*exp(Materials_price_index))

	xtabond Y_va K L, lags(1) maxldep(1) maxlags(1) vce(robust)

* BLUNDELL AND BOND (1999)

	xtdpdsys Y_va K L, lags(1) maxldep(1) maxlags(1) vce(robust)

* OLLEY AND PAKES (1996)
	
	* Check whether there are gaps for firms
	bysort firm_id (year): gen last_year = year[_N]
	bysort firm_id (year): gen first_year = year[1]
	bysort firm_id: gen has_gaps = (last_year - first_year + 1) != (_N)
	
	bysort firm_id: gen exit = (obs != 10) & (has_gaps == 0) & (year == last_year)
	replace exit = 0 if year == 1999
	
	opreg Y, exit(exit) state(K) proxy(I) free(L M) vce(bootstrap, seed(999) rep(250))

* LEVINSOHN AND PETRIN (2006)

	levpet Y_va, free(L) proxy(M) capital(K)

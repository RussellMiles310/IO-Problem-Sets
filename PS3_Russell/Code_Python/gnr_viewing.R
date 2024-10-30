#install.packages("gnrprod")
library(gnrprod)
View(gnrflex)
View(gnrprod::gauss_newton_reg)
library(readxl)
library(dplyr)

# Load the Excel file
filename <- "../PS3_data_changedtoxlsx.xlsx"
df0 <- read_excel(filename)

# Select specific columns and rename them
df <- df0 %>%
  select(year, firm_id, X03, X04, X05, X16, X40, X43, X44, X45, X49) %>%
  rename(t = year,
         firm_id = firm_id,
         y_gross = X03,
         s01 = X04,
         s02 = X05,
         s13 = X16,
         k = X40,
         l = X43,
         m = X44,
         py = X45,
         pm = X49)

# Drop rows where 'm' is 0 and filter for industry 1 only
df <- df %>%
  filter(m != 0, s13 == 1)

# Create new variables
df <- df %>%
  mutate(y = y_gross,
         s = pm + m - py - y)

# Sort by 'firm_id' and 't' for lagged variables
df <- df %>%
  arrange(firm_id, t) %>%
  group_by(firm_id) %>%
  mutate(kprev = lag(k),
         lprev = lag(l),
         mprev = lag(m)) %>%
  ungroup()

# View the final dataframe
print(df)

OUT = gnrflex(output = "y", fixed = c("l", "k"),
        flex = "m", share = "s", id = "firm_id",
        time = "t", data = df,
        control = list(degree = 2, maxit = 200))

plot(OUT$elas$residuals)

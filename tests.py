import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

def eleven(df ,to_select_columns):
    '''t-test of regressioin co-efficient
    '''
    x = df[to_select_columns[0]]
    y = df[to_select_columns[1]]

    x = sm.add_constant(x)

    model = sm.OLS(y, x)
    results = model.fit()

    coefficient_index = 1
    if np.abs(results.params[coefficient_index]) < 1e-8:
        p_value = 1.0
    else:
        # Perform the t-test for the regression coefficient
        t_value = results.tvalues[coefficient_index]
        p_value = results.pvalues[coefficient_index]


    # Set the significance level
    alpha = 0.05

    # Compare the p-value with the significance level
    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Reject the null hypothesis. The regression coefficient is statistically significant.")
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Fail to reject the null hypothesis. The regression coefficient is not statistically significant.")

def twelve(df,to_select_columns):
    X = df[to_select_columns[0]]  # targeted_productivity is 'independent_variable'
    y = df[to_select_columns[1]]  # actual_productivity is 'dependent_variable'

    # Perform Pearson correlation
    correlation, p_value = stats.pearsonr(X, y)

    # Set the significance level
    alpha = 0.05
    st.write("Correlation: ",correlation)
    # Compare the p-value with the significance level
    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Reject the null hypothesis. The correlation coefficient is statistically significant.")
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Fail to reject the null hypothesis. The correlation coefficient is not statistically significant.")

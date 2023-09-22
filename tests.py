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

def thirteen(df,to_select_columns):
    x = df[to_select_columns[0]]  # targeted_productivity is 'independent_variable'
    y = df[to_select_columns[1]]  # actual_productivity is 'dependent_variable'

    # Perform Pearson correlation
    correlation, _ = stats.pearsonr(x, y)

    # Compute the Fisher transformation
    fisher_transform = np.arctanh(correlation)

    # Calculate the standard error
    n = len(x)
    standard_error = 1 / np.sqrt(n - 3)

    # Set the significance level
    alpha = 0.05

    # Calculate the Z-score
    z_score = fisher_transform / standard_error

    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    st.write("Z value: ",z_score)

    # Compare the p-value with the significance level
    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Reject the null hypothesis. The correlation coefficient is statistically significant.")
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Fail to reject the null hypothesis. The correlation coefficient is not statistically significant.")

def fourteen(df,to_select_columns):

    x1 = df[to_select_columns[0]]  #  'independent_variable1'
    y1 = df[to_select_columns[1]]  #  'dependent_variable1'

    x2 = df[to_select_columns[2]]  #'independent_variable2'
    y2 = df[to_select_columns[3]]  #  'dependent_variable2'

    x1.dropna()
    x2.dropna()
    y1.dropna()
    y2.dropna()

    # Perform Pearson correlation
    correlation1, _ = stats.pearsonr(x1, y1)
    correlation2, _ = stats.pearsonr(x2, y2)

    # Compute the Fisher transformation for both correlations
    fisher_transform1 = np.arctanh(correlation1)
    fisher_transform2 = np.arctanh(correlation2)

    # Calculate the standard errors for both correlations
    n1 = len(x1)
    n2 = len(x2)
    standard_error1 = 1 / np.sqrt(n1 - 3)
    standard_error2 = 1 / np.sqrt(n2 - 3)

    # Calculate the Z-score for comparing the two correlations
    z_score = (fisher_transform1 - fisher_transform2) / np.sqrt(standard_error1**2 + standard_error2**2)
    st.write("Z value: ",z_score)
    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Set the significance level
    alpha = 0.05

    # Compare the p-value with the significance level
    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Reject the null hypothesis. The two correlation coefficients are significantly different.")
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write("Fail to reject the null hypothesis. The two correlation coefficients are not significantly different.")

import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin as pg


def eleven(df, to_selected_columns):
    """t-test of regressioin co-efficient"""
    x = df[to_selected_columns[0]]
    y = df[to_selected_columns[1]]

    x = sm.add_constant(x)

    model = sm.OLS(y, x)
    results = model.fit()

    coefficient_index = 1
    if np.abs(results.params[coefficient_index]) < 1e-8:
        p_value = 1.0
    else:
        t_value = results.tvalues[coefficient_index]
        p_value = results.pvalues[coefficient_index]

    alpha = 0.05

    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Reject the null hypothesis. The regression coefficient is statistically significant."
        )
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Fail to reject the null hypothesis. The regression coefficient is not statistically significant."
        )


def twelve(df, to_selected_columns):
    X = df[to_selected_columns[0]]
    y = df[to_selected_columns[1]]

    correlation, p_value = stats.pearsonr(X, y)

    alpha = 0.05
    st.write("Correlation: ", correlation)

    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Reject the null hypothesis. The correlation coefficient is statistically significant."
        )
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Fail to reject the null hypothesis. The correlation coefficient is not statistically significant."
        )


def thirteen(df, to_select_columns):
    x = df[to_select_columns[0]]
    y = df[to_select_columns[1]]

    correlation, _ = stats.pearsonr(x, y)

    fisher_transform = np.arctanh(correlation)

    n = len(x)
    standard_error = 1 / np.sqrt(n - 3)

    alpha = 0.05

    z_score = fisher_transform / standard_error

    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    st.write("Z value: ", z_score)

    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Reject the null hypothesis. The correlation coefficient is statistically significant."
        )
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Fail to reject the null hypothesis. The correlation coefficient is not statistically significant."
        )


def fourteen(df, to_select_columns):
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
    z_score = (fisher_transform1 - fisher_transform2) / np.sqrt(
        standard_error1**2 + standard_error2**2
    )
    st.write("Z value: ", z_score)
    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Set the significance level
    alpha = 0.05

    # Compare the p-value with the significance level
    if p_value < alpha:
        st.write(f"F-statistic: {f_statistic:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Reject the null hypothesis. There are significant differences in the population means."
        )
    else:
        st.write(f"F-statistic: {f_statistic:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Fail to reject the null hypothesis. There is no significant difference in the population means."
        )


def seventeen(df, to_selected_columns):
    sample_data1 = df[to_selected_columns[0]]
    sample_data2 = df[to_selected_columns[1]]

    sample_data1 = sample_data1.dropna()
    sample_data2 = sample_data2.dropna()

    sample_variance1 = np.var(sample_data1, ddof=1)
    sample_variance2 = np.var(sample_data2, ddof=1)
    df1 = len(sample_data1) - 1
    df2 = len(sample_data2) - 1

    f_stat = sample_variance1 / sample_variance2

    critical_value = stats.f.ppf(0.95, df1, df2)

    alpha = 0.05

    # Compare the F-statistic with the critical value
    if f_stat < critical_value:
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"Critical value F({df1}, {df2}; {alpha}): {critical_value:.4f}")
        st.write(
            "Do not reject the null hypothesis. The variances are not significantly different."
        )
    else:
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"Critical value F({df1}, {df2}; {alpha}): {critical_value:.4f}")
        st.write(
            "Reject the null hypothesis. The variances are significantly different."
        )


def sixteen(df, to_selected_columns):
    # Extract the relevant columns for the variance analysis
    sample_data1 = df[
        to_selected_columns[0]
    ]  # Replace 'column_name1' with the appropriate column name for the first sample
    sample_data2 = df[
        to_selected_columns[1]
    ]  # Replace 'column_name2' with the appropriate column name for the second sample

    # Remove NaN values from both samples, if any
    sample_data1 = sample_data1.dropna()
    sample_data2 = sample_data2.dropna()

    # Calculate the sample variances and degrees of freedom for both samples
    sample_variance1 = sample_data1.var()
    sample_variance2 = sample_data2.var()
    df1 = len(sample_data1) - 1
    df2 = len(sample_data2) - 1

    # Perform the F-test
    f_stat = sample_variance1 / sample_variance2

    # Calculate the critical value for F at a 5% significance level and degrees of freedom (df1, df2)
    critical_value = stats.f.ppf(0.95, df1, df2)

    # Set the significance level
    alpha = 0.05

    # Compare the F-statistic with the critical value
    if f_stat < critical_value:
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"Critical value F({df1}, {df2}; {alpha}): {critical_value:.4f}")
        st.write(
            "Do not reject the null hypothesis. The variances are not significantly different."
        )
    else:
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"Critical value F({df1}, {df2}; {alpha}): {critical_value:.4f}")
        st.write(
            "Reject the null hypothesis. The variances are significantly different."
        )


def twenty_two(df, to_selected_columns):
    # Extract the relevant columns for the ANOVA
    # variable_to_test = 'actual_productivity'  # Replace 'actual_productivity' with the column name representing the variable you want to test
    # to_selected_columns[0] = 'quarter'  # Replace 'quarter' with the column name representing the groups (e.g., 'Quarter1', 'Quarter2', etc.)

    # Perform the one-way ANOVA
    groups = df[to_selected_columns[0]].unique()
    anova_results = stats.f_oneway(
        *(
            df[df[to_selected_columns[0]] == group][to_selected_columns[1]]
            for group in groups
        )
    )

    # Get the F-statistic and p-value from the ANOVA results
    f_statistic = anova_results.statistic
    p_value = anova_results.pvalue
    # Calculate the Z-score
    z_score = fisher_transform / standard_error

    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    st.write("Z value: ", z_score)

    # Compare the p-value with the significance level
    if p_value < alpha:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Reject the null hypothesis. The correlation coefficient is statistically significant."
        )
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Fail to reject the null hypothesis. The correlation coefficient is not statistically significant."
        )


def twenty_four(df, to_selected_columns):
    st.write("Select only one column")
    assumed_variance = st.number_input("Enter the assumed population variance")
    # Extract the relevant column for the test
    variable_to_test = df[
        to_selected_columns[0]
    ]  # Replace 'targeted_productivity' with the name of the variable you want to test

    # Assumed population variance

    # Calculate the sample variance
    sample_variance = np.var(variable_to_test, ddof=1)

    # Calculate the Chi-square test statistic
    chi2_statistic = (len(variable_to_test) - 1) * sample_variance / assumed_variance

    # Degrees of freedom for the Chi-square distribution
    dof = len(variable_to_test) - 1

    # Calculate the critical value from the Chi-square distribution
    critical_value = stats.chi2.ppf(
        0.95, dof
    )  # You can change the significance level (0.95) if needed

    # Compare the test statistic with the critical value
    if chi2_statistic < critical_value:
        st.write(f"Chi-square test statistic: {chi2_statistic:.4f}")
        st.write(f"Critical value: {critical_value:.4f}")
        st.write(
            "Do not reject the null hypothesis. The population variance matches the assumed value."
        )
    else:
        st.write(f"Chi-square test statistic: {chi2_statistic:.4f}")
        st.write(f"Critical value: {critical_value:.4f}")
        st.write(
            "Reject the null hypothesis. The population variance does not match the assumed value."
        )


def twenty_six(df, to_selected_columns):
    # Extract the relevant columns for the ANOVA
    variable_to_test = to_selected_columns[
        0
    ]  # Replace 'actual_productivity' with the column name representing the variable you want to test
    group_variable = to_selected_columns[
        1
    ]  # Replace 'department' with the column name representing the groups (e.g., 'sweing', 'finishing', etc.)

    # Perform the one-way ANOVA
    groups = df[group_variable].unique()
    anova_results = stats.f_oneway(
        *(df[df[group_variable] == group][variable_to_test] for group in groups)
    )

    # Get the F-statistic and p-value from the ANOVA results
    f_statistic = anova_results.statistic
    p_value = anova_results.pvalue

    # Set the significance level
    alpha = 0.05

    # Compare the p-value with the significance level
    if p_value < alpha:
        st.write(f"F-statistic: {f_statistic:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Reject the null hypothesis. There are significant differences in the subpopulation means."
        )
    else:
        st.write(f"F-statistic: {f_statistic:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Fail to reject the null hypothesis. There is no significant difference in the subpopulation means."
        )


def thirty_one(df, to_selected_columns):
    # Extract the relevant columns for Bartlett's test
    variable_to_test = to_selected_columns[
        0
    ]  # Replace 'actual_productivity' with the column name representing the variable you want to test
    group_variable = to_selected_columns[
        1
    ]  # Replace 'department' with the column name representing the groups (e.g., 'sweing', 'finishing', etc.)

    # Create a list of data for each department
    group_data = [
        df[df[group_variable] == group][variable_to_test]
        for group in df[group_variable].unique()
    ]

    # Perform Bartlett's test for equality of K variances
    bartlett_statistic, p_value = stats.bartlett(*group_data)

    # st.write the results
    st.write(f"Bartlett's test statistic: {bartlett_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")


def thirty_two(df, to_selected_columns):
    # Extract the relevant columns for Hartley's test
    variable_to_test = to_selected_columns[
        0
    ]  # Replace 'actual_productivity' with the column name representing the variable you want to test
    group_variable = to_selected_columns[
        1
    ]  # Replace 'department' with the column name representing the groups (e.g., 'sweing', 'finishing', etc.)

    # Create a list of data for each department
    group_data = [
        df[df[group_variable] == group][variable_to_test]
        for group in df[group_variable].unique()
    ]

    # Calculate the largest variance and the smallest variance
    max_variance = max([variance.var() for variance in group_data])
    min_variance = min([variance.var() for variance in group_data])

    # Calculate Hartley's F-ratio
    hartley_f_ratio = max_variance / min_variance

    # Get the degrees of freedom for numerator and denominator
    numerator_df = len(group_data) - 1
    denominator_df = len(df) - len(group_data)

    # Get the critical value for Hartley's F-ratio at a given significance level (e.g., 0.05)
    critical_value = stats.f.ppf(0.95, numerator_df, denominator_df)
    st.write("Hartley's F-ratio: ", hartley_f_ratio)
    st.write("Critical value: ", critical_value)
    # Perform Hartley's test
    if hartley_f_ratio < critical_value:
        st.write("Hartley's test: The variances are approximately equal.")
    else:
        st.write("Hartley's test: The variances are not equal.")


def thirty_three(df, to_selected_columns):
    # Extract the relevant column for the Shapiro-Wilk test
    variable_to_test = to_selected_columns[
        0
    ]  # Replace 'actual_productivity' with the column name representing the variable you want to test

    # Get the data for the Shapiro-Wilk test
    data_for_test = df[variable_to_test]

    # Perform the Shapiro-Wilk test for normality
    statistic, p_value = stats.shapiro(data_for_test)

    # st.write the results
    st.write(f"Shapiro-Wilk test statistic: {statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")

    # Check if the data is normally distributed based on the p-value
    if p_value > 0.05:
        st.write("The data appears to be normally distributed.")
    else:
        st.write("The data does not appear to be normally distributed.")


def thrity_four(df, to_selected_columns):
    # Extract the relevant columns for Cochran's test
    variable_to_test = to_selected_columns[
        0
    ]  # Replace 'actual_productivity' with the column name representing the variable you want to test
    group_variable = to_selected_columns[
        1
    ]  # Replace 'department' with the column name representing the groups (e.g., 'sweing', 'finishing', etc.)

    # Create a list of data for each department
    group_data = [
        df[df[group_variable] == group][variable_to_test]
        for group in df[group_variable].unique()
    ]

    # Calculate Cochran's C statistic
    num_groups = len(group_data)
    n_values = [len(group) for group in group_data]
    sorted_n_values = sorted(n_values, reverse=True)
    largest_n = sorted_n_values[0]
    C_statistic = (largest_n - 1) / (sum([(1 / n) for n in n_values]) - 1)

    # Get the critical value for Cochran's C statistic at a given significance level (e.g., 0.05)
    critical_value = stats.chi2.ppf(0.95, num_groups - 1)
    st.write(" Cohran's C_statistic: ", C_statistic)
    st.write("Critical Value: ", critical_value)
    # Perform Cochran's test
    if C_statistic > critical_value:
        st.write("Cochran's test: There are variance outliers among the groups.")
    else:
        st.write("Cochran's test: There are no variance outliers among the groups.")


def thirty_five(df, to_selected_columns):
    # Extract the relevant column for the Kolmogorov-Smirnov test
    variable_to_test = to_selected_columns[
        0
    ]  # Replace 'actual_productivity' with the column name representing the variable you want to test

    # Get the data for the Kolmogorov-Smirnov test
    data_for_test = df[variable_to_test]

    # Perform the Kolmogorov-Smirnov test for goodness of fit
    # Here, we are testing against the normal distribution, but you can replace 'norm' with other distributions like 'expon', 'uniform', etc.
    # You may also specify additional distribution parameters (e.g., mean and standard deviation for the normal distribution).
    # The 'args' parameter is used to pass the distribution parameters.
    statistic, p_value = stats.kstest(
        data_for_test, "norm", args=(data_for_test.mean(), data_for_test.std())
    )

    # st.write the results
    st.write(f"Kolmogorov-Smirnov test statistic: {statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")

    # Check if the data follows the given distribution based on the p-value
    if p_value > 0.05:
        st.write("The data follows the normal distribution.")
    else:
        st.write("The data does not follow the normal distribution.")


def thirty_six(df, to_selected_columns):
    # Extract the relevant columns for the Kolmogorov-Smirnov test
    variable_to_compare = to_selected_columns[
        0
    ]  # Replace 'actual_productivity' with the column name representing the variable you want to compare
    group_variable = to_selected_columns[
        1
    ]  # Replace 'department' with the column name representing the groups (e.g., 'sweing', 'finishing', etc.)

    # Get the data for the two populations to compare
    population_1_data = df[df[group_variable] == "sweing"][variable_to_compare]
    population_2_data = df[df[group_variable] == "finishing"][variable_to_compare]

    # Perform the Kolmogorov-Smirnov test for comparing two populations
    statistic, p_value = stats.ks_2samp(population_1_data, population_2_data)

    # st.write the results
    st.write(f"Kolmogorov-Smirnov test statistic: {statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")

    # Check if the two populations have the same distribution based on the p-value
    if p_value > 0.05:
        st.write("The two populations have the same distribution.")
    else:
        st.write("The two populations do not have the same distribution.")


def thirty_seven(df, to_selected_columns):
    # Extract the relevant column for the Chi-Square test
    variable_to_test = to_selected_columns[
        0
    ]  # Replace 'department' with the column name representing the variable you want to test

    # Get the observed frequencies
    observed_frequencies = df[variable_to_test].value_counts()

    # If the data contains NaN values, replace them with 0
    observed_frequencies = observed_frequencies.fillna(0)

    # Perform the Chi-Square test for goodness of fit
    expected_frequencies = np.full(
        len(observed_frequencies), len(df) / len(observed_frequencies)
    )
    chi_square_statistic, p_value = stats.chisquare(
        observed_frequencies, f_exp=expected_frequencies
    )

    # st.write the results
    st.write(f"Chi-Square test statistic: {chi_square_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")

    # Check if the observed frequencies fit the expected frequencies based on the p-value
    if p_value > 0.05:
        st.write("The observed frequencies fit the expected frequencies.")
    else:
        st.write("The observed frequencies do not fit the expected frequencies.")


def thirty_eight(df, to_selected_columns):
    # Extract the relevant columns for the Chi-Square test
    variable_1 = to_selected_columns[
        0
    ]  # Replace 'department' with the first column name representing the variable you want to test
    variable_2 = to_selected_columns[
        1
    ]  # Replace 'quarter' with the second column name representing the variable you want to test

    # Create a contingency table
    contingency_table = pd.crosstab(df[variable_1], df[variable_2])

    # Perform the Chi-Square test for compatibility of K counts
    chi_square_statistic, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )

    # st.write the results
    st.write(f"Chi-Square test statistic: {chi_square_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    st.write(f"Degrees of freedom: {dof}")
    st.write("Expected frequencies:")
    st.write(expected)

    # Check if there is a significant association between the variables based on the p-value
    if p_value > 0.05:
        st.write("There is no significant association between the variables.")
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Reject the null hypothesis. The two correlation coefficients are significantly different."
        )
    else:
        st.write(f"p-value: {p_value:.4f}")
        st.write(
            "Do not reject the null hypothesis. The two correlation coefficients are not significantly different."
        )


def fourty(df, to_selected_columns):
    variable_1 = to_selected_columns[0]
    variable_2 = to_selected_columns[1]
    contingency_table = pd.crosstab(df[variable_1], df[variable_2])
    chi_square_statistic, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )
    st.write(f"Chi-Square test statistic: {chi_square_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    st.write(f"Degrees of freedom: {dof}")
    st.write("Expected frequencies:")
    st.write(expected)
    if p_value > 0.05:
        st.write("There is no significant association between the variables.")
    else:
        st.write("There is a significant association between the variables.")


def fourty_one(df, to_selected_columns):
    categorical_variable = to_selected_columns[0]
    binary_variable = to_selected_columns[1]
    contingency_table = pd.crosstab(df[categorical_variable], df[binary_variable])
    chi_square_statistic, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )
    st.write(f"Chi-Square test statistic: {chi_square_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    st.write(f"Degrees of freedom: {dof}")
    st.write("Expected frequencies:")
    st.write(expected)
    if p_value > 0.05:
        st.write("There is no significant association between the variables.")
    else:
        st.write("There is a significant association between the variables.")


def fourty_two(df, to_selected_columns):
    n = to_selected_columns[0]
    k = to_selected_columns[1]
    crosstab_data = pd.crosstab(df[n], df[k])
    result = pg.cochran(crosstab_data)
    st.write(result)


def fourty_three(df, to_selected_columns):
    binary_variable = to_selected_columns[0]
    categorical_variable = to_selected_columns[1]
    contingency_table = pd.crosstab(df[binary_variable], df[categorical_variable])
    chi_square_statistic, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )
    st.write(f"Chi-Square test statistic: {chi_square_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    st.write(f"Degrees of freedom: {dof}")
    st.write("Expected frequencies:")
    st.write(expected)
    if p_value > 0.05:
        st.write("There is no significant association between the variables.")
    else:
        st.write("There is a significant association between the variables.")

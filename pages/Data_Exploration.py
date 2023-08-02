import streamlit as st
import pandas as pd
import util
import tests
from test_dict import test_dict

def main():
    tests_df = pd.read_csv("Sheet1.csv")
    with st.sidebar:
        tests,_ = util.filter_dataframe(tests_df)
    try:
        for test in tests.itertuples():
            with st.expander(f"{test.Test_Name}"):
                df,to_filter_columns = util.filter_dataframe(st.session_state.df,max=4,key=test.Test_Name)
                df = df[list(to_filter_columns)]
                st.dataframe(df)
                try:
                 func = test_dict[test.Test_Name]
                 func(df,to_filter_columns)
                except Exception as e:
                    print(e)
                    pass
    except AttributeError as e:
        st.header("No file uploaded yet!")
        st.write("Upload dataframe and try again")

if __name__ == '__main__':
    util.page_config("Data Exploration")
    util.add_bg()
    util.add_title()
    main()

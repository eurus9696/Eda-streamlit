import streamlit as st
import pandas as pd
import util
import tests
from test_dict import test_dict


def main():
    tests_df = pd.read_csv("Sheet1.csv")
    with st.sidebar:
        tests, _ = util.filter_dataframe(tests_df, key="HI")
    test = st.selectbox(
        "Select your test here",
        tests["Test_Name"],
        placeholder="Select test...",
    )
    with st.expander(f"{test}"):
        df, to_filter_columns = util.filter_dataframe(
            st.session_state.df, max=4, key=test
        )
        df = df[list(to_filter_columns)]
        st.dataframe(df)
        try:
            func = test_dict[test]
            try:
                func(df, to_filter_columns)
            except Exception as e:
                pass
        except:
            pass


if __name__ == "__main__":
    util.page_config("Statistical Tests")
    util.add_bg()
    util.add_title()
    main()

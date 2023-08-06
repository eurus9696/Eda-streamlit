import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def read_file(uploaded_file):
    '''Read a csv/xlxs file and return a pandas dataframe
    '''
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        return None
    return df

def get_nums(data_frame: pd.DataFrame):
    return data_frame.select_dtypes(include = 'number')

def get_cats(data_frame):
    return data_frame.select_dtypes(include = 'object')

def page_config(title):
    st.set_page_config(
        page_title=title,
        menu_items={
            "Get Help":"https://www.github.com/eurus9696"
            },
        page_icon="./sas-removebg-preview.png",
        layout="wide",
        initial_sidebar_state='expanded'
    )

def add_title():
    '''Insert title before the navigation area in
    the sidebar
    '''
    st.markdown(
    """
        <style>
        [data-testid="stSidebarNav"]::before {
                content: "EDA";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 35px;
                font-weight:bold;
                position: relative;
                top: 100px;
            }
        </style>
     """,
        unsafe_allow_html=True,
    )

def add_bg():
    '''Add the background image to each page
    '''
    st.markdown(
            """
            <style>
            .stApp {
                background-image: url('http://getwallpapers.com/wallpaper/full/d/1/7/202392.jpg');
                background-size: cover;
                background-position: center;
                min-height: 100vh;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

def filter_dataframe(df: pd.DataFrame,max:int =None,key:str = None) -> pd.DataFrame:

    if max == "k":
        max = None
    df = df.copy()
    #Converting datetimes
    for col in df.columns:
    #    if is_object_dtype(df[col]):
    #        try:
    #            df[col] = pd.to_datetime(df[col])
    #        except Exception:
    #            pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    #Creating string to be displayed on the selectbox
    if max == 1:
        string = f"(choose 1 column)"
    elif max == 2:
        string = f"(choose 2 columns)"
    else:
        string = f"(choose 2 or more columns)"
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect(f"Filter dataframe on {string}", df.columns,max_selections=max,key=key)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
    return df,to_filter_columns

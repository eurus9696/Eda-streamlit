import pandas as pd
import streamlit as st

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        return None
    return df

def var_selectboxes(label,columns,help=None,exclude_list=None):
    if exclude_list is not None:
        columns = columns.difference(exclude_list)
    select_val = st.selectbox(label,columns,help=help)
    return select_val

def get_nums(data_frame):
    return data_frame.select_dtypes(include = 'number')

def get_cats(data_frame):
    return data_frame.select_dtypes(include = 'object')

def add_title():
    st.markdown(
    """
        <style>
        [data-testid="stSidebarNav"]::before {
                content: "Tools";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
     """,
        unsafe_allow_html=True,
    )

def add_bg():
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

import streamlit as st
# import streamlit.components.v1 as components
# import pygwalker as pyg
# import pandas as pd
import util



def render_main_page():
    # First row: Image, Name, Image
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("./sas-removebg-preview.png")

    with col2:
        st.title("EDA")

    with col3:
        st.image("./digiq.jpg")

    st.caption('''
                Discover the hidden potential within your datasets using [Streamlit-EDA]. Our user-friendly EDA and visualization website empowers you to analyze
               data effortlessly. Dive deep into patterns, trends, and correlations, presenting them visually with interactive charts and graphs.
               Collaborate with your team, share insights, and make data-driven decisions with confidence. Whether you're a seasoned analyst or a beginner,
               [Streamlit-EDA] equips you with the tools to explore data comprehensively and gain valuable knowledge from it.
               Sign up now to embark on a transformative journey of data exploration, gaining unparalleled understanding of your information. ''')

    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.session_state.uploaded = True
        st.session_state.df = util.read_file(uploaded_file)
        st.session_state.nums = util.get_nums(st.session_state.df)
        st.session_state.cats = util.get_cats(st.session_state.df)
        st.session_state.c_names = st.session_state.df.columns

    if 'button' not in st.session_state:
        st.session_state.button = False

    def click_button():
        st.session_state.button = not st.session_state.button
    st.button("Show data",on_click=click_button,disabled= not st.session_state.uploaded)
    if st.session_state.button:
        st.write(st.session_state.df)


def main():
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    render_main_page()

if __name__ == "__main__":
    util.page_config("EDA")
    util.add_title()
    util.add_bg()
    main()

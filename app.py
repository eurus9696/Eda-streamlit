import streamlit as st
import streamlit.components.v1 as components
import pygwalker as pyg
import pandas as pd
import util

st.set_page_config(
        page_title="EDA",
        menu_items={
            "Get Help":"https://www.github.com/eurus9696"
            },
        page_icon="./sas-removebg-preview.png",
        layout="wide",
        initial_sidebar_state='expanded'
    )



def pygwalker(df):
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

    st.title('Visualisation')
    components.html("", height=0)
    pyg_html=str(pyg.walk(df,return_html=True,login_required=False))
    components.html(pyg_html,height=1080,scrolling=True)

def render_main_page():
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

    # First row: Image, Name, Image
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("./sas-removebg-preview.png")

    with col2:
        st.title("EDA")

    with col3:
        st.image("./digiq.jpg")

    st.caption('''Lorem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit Lorem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit.orem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit.orem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit.orem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit.orem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit.orem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit.orem
               ipsum dolor sit
               amet, consectetur
               adipiscing elit..''')

    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.session_state.df = util.read_file(uploaded_file)
        st.session_state.nums = util.get_nums(st.session_state.df)
        st.session_state.cats = util.get_cats(st.session_state.df)
        st.session_state.c_names = st.session_state.df.columns
    if st.button("Show data"):
        try:
            st.write(st.session_state.df)
        except AttributeError:
            st.header("No dataframe uploaded")
            st.write("Upload dataframe and try again")
    a = util.var_selectboxes("Choose a column",st.session_state.c_names)
    st.write(a)


def main():
    st.sidebar.title("Tools")
    menu = ["EDA", "Visualisations"]
    choice = st.sidebar.radio("Go to", menu)

    if choice == "EDA":
        render_main_page()
    elif choice == "Visualisations":
        try:
            pygwalker(st.session_state.df)
        except AttributeError:
            st.header("No dataframe uploaded yet")
            st.write("Upload dataframe and try again")
if __name__ == "__main__":
    main()

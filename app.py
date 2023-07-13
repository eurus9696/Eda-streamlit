import streamlit as st
import streamlit.components.v1 as components
import pygwalker as pyg
import pandas as pd
st.set_page_config(
        page_title="EDA",
        menu_items={
            "Get Help":"https://www.github.com/eurus9696"
            },
        page_icon="./sas-removebg-preview.png",
        layout="wide",
)
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        return None
    return df

def pygwalker(df):
    components.html("", height=0)
    pyg_html=str(pyg.walk(df,return_html=True,login_required=False))
    components.html(pyg_html,scrolling=True)

def main():
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
        st.image("./sas-removebg-preview.png")

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
        df = read_file(uploaded_file)
        st.write(df)
        if st.button("Go to Pygwalker"):
            pygwalker(df)

if __name__ == "__main__":
    main()

    

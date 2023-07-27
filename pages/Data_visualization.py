import streamlit as st
import pygwalker
import streamlit.components.v1 as components
import util

def pygwalker(df):
    st.title('Visualisation')
    components.html("", height=0)
    pyg_html=str(pyg.walk(df,return_html=True,login_required=False))
    components.html(pyg_html,height=1080,scrolling=True)

def main():
    util.add_title()
    util.add_bg()
    try:
        pygwalker(st.session_state.df)
    except AttributeError:
        st.header("No file uploaded yet!")
        st.write("Upload file and try again")
if __name__ == '__main__':
    main()

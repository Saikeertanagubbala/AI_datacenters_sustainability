import streamlit as st

st.title("My Streamlit App")
st.write("Hello, welcome to my app!")

tab1, tab2, tab3, tab4 = st.tabs(["Pre-processing steps", "ISA", "EDA", "Documentation"], width=1000)



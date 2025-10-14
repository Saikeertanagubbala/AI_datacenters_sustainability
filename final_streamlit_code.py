import streamlit as st
st.set_page_config(layout="wide")


st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        padding: 16px 24px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(":earth_americas: Data Centers and Non-Sustainability Measures :fire:")
st.write("Learn more about metrics on data centers and their usage of energy and water, pushing for sustainability moving forward.")

tab1, tab2, tab3, tab4 = st.tabs(["Pre-processing steps", "IDA", "EDA", "Documentation"])

with tab1:
    st.write("Pre-processing content here")

with tab2:
    st.write("IDA content here")

with tab3:
    st.write("EDA content here")

with tab4:
    st.write("The two datasets were from Kaggle.")
    url1 = "https://www.kaggle.com/datasets/sudalairajkumar/data-centers-energy-and-water-usage"
    url2 = "https://www.kaggle.com/datasets/anmolkumar/2023-esg-metrics-dataset"
    multi = '''Dataset 1 (ESG Metrics): [link](%s)
Dataset 2 (Data Center Metrics): 

Two (or more) newline characters in a row will result in a hard return.
'''
    st.markdown(multi)

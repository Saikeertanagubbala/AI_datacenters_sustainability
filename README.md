# Data Centers & ESG 

## Project overview
This project explores the environmental impact of data centers and their relationship with company ESG metrics. It combines a country-level data centers dataset with a company-level ESG dataset to study trends in carbon emissions, water usage, and energy consumption.

## Repository structure
- final_streamlit_code.py: Main Streamlit app with 3 tabs for documentation, IDA and EDA.
- final_eda_plots.ipynb: Notebook for exploratory plots which were intended to go for the EDA plots.
- Part 1 - 4 cleaning .ipynb's: These files go in order strictly for cleaning the original csv files, mainly directed towards the data centers csv file. They go through the process of cleaning numerical columns, categorical columns, then impuattion, correlation matrices, as well as skewness. 

## Why this dataset
I initially wanted to explore thsi topic due to the growing concern of the amount of water AI uses. However, there are little datasets with this information, and many tech companies tend to keep information pertaining to environmental metrics hard to find. I was able to find a synthetic dataset, along with another that scraped multiple online sources. With rapid AI adoption, data center counts and capacities are growing by year. USing my datasets, I wanted to quantify the environmental impact strictly for water usage for cooling and carbon emissions from power. This makes the intersection of data centers and ESG metrics a meaningful place to study sustainability risks and trends.

## What I learned from IDA / EDA
- Missingness: The data centers dataset contained decent amount of missing values across both numerical and categorical features; the ESG dataset was largely synthetic/complete. The data centers dataset was scraped from websites online making the values show up as text rather than numerical for the most part. I've had to learn how to deal with special characters, ranges, and incorrect values. 
- Distribution & skew: Many numerical features (e.g., power capacity, floor space, renewable usage percentage) are skewed; log transforms helped reduce skewness for visualization and modeling.
- Correlations: After imputation and transformations, sensible correlations emerged (like how larger power capacity often correlates with higher carbon emissions).
- Time trends: Regional time series show increases in carbon emissions, water usage, and energy consumption over recent years, with North America and Asia Pacific often leading.

## Preprocessing steps completed
- Missing value handling:
  - SimpleImputer (median) for heavily skewed variables.
  - KNNImputer (hybrid approach) for other variables where SImpleImputer didn't do as well.
- Transformations:
  - Log transform for skewed variables (with small shifts for non-positive values).
- Subsetting:
  - Extracted numeric-only subsets for correlation and skewness analyses.
  - Focused the ESG analysis on the Technology industry for relevance to data centers.
- Feature selection and cleaning:
  - Dropped irrelevant identifiers and non-informative columns for plotting and correlation analysis.

## What I tried with Streamlit so far
- Used a multi-tab feature to organize narrative, plots, and analysis.
- Used minimal CSS to make tab text bigger and change the color. 
- Interactive visualizations:
  - Time-series (plotly) of Carbon Emissions / Water Usage / Energy Consumption by region. Borrowed this from some team-mates who created the feature to show the same graph but with different variables showing. 
  - Scatter plots showing power capacity vs carbon emissions with hover details (plotly feature).
- Interactive controls:
  - Selectboxes to choose metrics to plot.
  - Expanders to collapse/expand detailed IDA sections.
  - Multiple-tabs at the top labelled as "Documentation", " IDA", "EDA"

## How to run
1. Create a Python environment and install dependencies:
   - pandas, numpy, matplotlib, seaborn, scikit-learn, plotly, streamlit
2. Start the Streamlit app:
   - streamlit run final_streamlit_code.py
3. Separate tab should open up with the stremalit app.
4. The individual .ipynb files can also be run if you'd like to see it in a notebook.


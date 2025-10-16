import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    data_centers = pd.read_csv("data_centers.csv", 
                               usecols=['country', 'total_data_centers', 'hyperscale_data_centers',
                                       'colocation_data_centers', 'floor_space_sqft_total',
                                       'power_capacity_MW_total', 'average_renewable_energy_usage_percent',
                                       'tier_distribution', 'key_operators', 'cloud_provider',
                                       'internet_penetration_percent','growth_rate_of_data_centers_percent_per_year',
                                       'cooling_technologies_common'],
                               na_values=['N/A', 'NA', 'unknown', '', None, 'Unknown', '?', 'No data'])
    return data_centers

data_centers = load_data()
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        padding: 16px 24px !important;
        font-weight: 600 !important;
        color: #4A90E2 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #ff4b4b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(":earth_americas: Data Centers and Non-Sustainability Measures :fire:")
st.write("Learn more about metrics on data centers and their usage of energy and water, pushing for sustainability moving forward.")

tab1, tab2, tab3 = st.tabs(["Documentation", "IDA", "EDA"])

with tab2:
    with st.expander("### Data Centers: Missing Values Analysis", expanded=True):
        st.write("- My missing values centered around my 'Data Centers' dataset, compared to my ESG dataset which is synthetic and complete.")
        st.write("- The data ceneters dataset had missing values for both numerical and categrical variables.")
        st.write("- The ESG dataset had missing values for a variable that I ended up dropping due to irrelevance to my analysis.")

        col1, col2 = st.columns([2, 1]) 
        
        # Left column: Heatmap
        with col1:
            nan_mask = data_centers.isna()
            nan_array = nan_mask.astype(int).to_numpy()
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')
            ax.set_xlabel('Range Index (len of df)', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            ax.set_title('Visualizing Missing Values in Data Centers Dataset', fontsize=14, fontweight='bold')
            
            ax.set_yticks(range(len(data_centers.columns)))
            ax.set_yticklabels(data_centers.columns, fontsize=10)
            
            data_centers_values = nan_array.shape[0]
            ax.set_xticks(np.linspace(0, data_centers_values-1, 
                                       min(10, data_centers_values)).astype(int))
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**Missing Values Summary**")
            
            summary_data = []
            for col in data_centers.columns:
                n_missing = data_centers[col].isna().sum()
                n_total = len(data_centers[col])
                pct_missing = (n_missing / n_total) * 100
                
                if n_missing > 0:  # Only show columns with missing values
                    summary_data.append({
                        'Feature': col,
                        'Missing': n_missing,
                        'Missing %': f"{pct_missing:.1f}%"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Missing', ascending=False)
            
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("### Data Centers: Skewness Analysis", expanded=True):
        st.markdown("We look for skewness closer to 0, indicating a more symmetric distribution. Values >1 or <-1 indicate high skewness.")
    
        df_cleaned = pd.read_csv("cleaned_numeric_data_centers.csv")
        skewness_data = {
            'Feature': [
                'total_data_centers',
                'hyperscale_data_centers',
                'colocation_data_centers',
                'internet_penetration_percent',
                'growth_rate_of_data_centers_percent_per_year',
                'floor_space_sqft_total',
                'power_capacity_MW_total',
                'average_renewable_energy_usage_percent'
            ],
            'Skewness': [
                df_cleaned['total_data_centers'].skew(),
                df_cleaned['hyperscale_data_centers'].skew(),
                df_cleaned['colocation_data_centers'].skew(),
                df_cleaned['internet_penetration_percent'].skew(),
                df_cleaned['growth_rate_of_data_centers_percent_per_year'].skew(),
                df_cleaned['floor_space_sqft_total'].skew(),
                df_cleaned['power_capacity_MW_total'].skew(),
                df_cleaned['average_renewable_energy_usage_percent'].skew()
            ]
        }
        
        skewness_df = pd.DataFrame(skewness_data)
        skewness_df['Skewness'] = skewness_df['Skewness'].round(3)
        st.dataframe(skewness_df, use_container_width=True)

    with st.expander("### Data Centers: Imputation", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("This correlation heatmap will drop all the missing values and possibily show higher correlations amongst variables pre-maturely.")
            df = pd.read_csv("cleaned_numeric_data_centers.csv")
            numerical_cols_with_missing = [
            'floor_space_sqft_total', 
            'power_capacity_MW_total', 
            'average_renewable_energy_usage_percent', 
            'internet_penetration_percent',
            'growth_rate_of_data_centers_percent_per_year'
            ]
            complete_numerical_cols = [
                'total_data_centers', 
                'hyperscale_data_centers', 
                'colocation_data_centers'
            ]

            all_numerical_cols = complete_numerical_cols + numerical_cols_with_missing
            data_complete = df[all_numerical_cols].dropna()
            corr_before = data_complete.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_before, annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0)
            ax.set_title('Before Imputation')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close() 
        with col2:
            st.markdown("Post impuation and transformation, we can see more realistic correlations between variables.")
            data_centers = pd.read_csv("data_centers_numeric_imputed.csv", usecols=['country', 'total_data_centers', 'hyperscale_data_centers',
                                                        'colocation_data_centers', 'floor_space_sqft_total',
                                                        'power_capacity_MW_total', 'average_renewable_energy_usage_percent',
                                                        'tier_distribution',
                                                        'internet_penetration_percent','growth_rate_of_data_centers_percent_per_year',
                                                        'cooling_technologies_common'])

            cols = ['total_data_centers', 'hyperscale_data_centers', 'colocation_data_centers',
                    'internet_penetration_percent', 'growth_rate_of_data_centers_percent_per_year', 'floor_space_sqft_total',
                    'power_capacity_MW_total', 'average_renewable_energy_usage_percent']
            correlation = data_centers[cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
            ax.set_title("Data Center Features (imputed (KNN + simple imputer) & log transformed)")
            st.pyplot(fig)
            plt.close() 
#-------------------------------------------------------------------Simple imputer plot -------------------------------------------------------------
        col1, col2 = st.columns([1, 1])
        with col1:
            median_imputer = SimpleImputer(strategy='median', missing_values=np.nan)
            X_missing = df[numerical_cols_with_missing]
            imputed_X = median_imputer.fit_transform(X_missing)

            imputed_df = pd.DataFrame(imputed_X, columns=numerical_cols_with_missing, index=df.index)
            data_centers_imputed = df.copy()
            data_centers_imputed[numerical_cols_with_missing] = imputed_df

            #------------------------FIGURES--------------------------------------------------
            fig, axes = plt.subplots(3, 2, figsize=(10, 8)) # 5 plots for 5 missing variables
            axes = axes.flatten()

            for idx, col in enumerate(numerical_cols_with_missing):
                ax = axes[idx]
                
                # Original data with missing values
                original_data = df[col].dropna()
                
                # Complete imputed dataset (NaNs replaced with median of column)
                imputed_complete_data = data_centers_imputed[col]
                
                # Plot original data
                ax.hist(original_data, bins=30, alpha=0.5, label=f'Original (n={len(original_data)})', 
                        color='blue')
                
                # Plot complete imputed data on top
                ax.hist(imputed_complete_data, bins=30, alpha=0.5, label=f'After Imputation (n={len(imputed_complete_data)})', 
                        color='red')
                
                ax.set_title(f'{col}', fontsize=10)
                ax.set_xlabel('Values')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)

            fig.delaxes(axes[5])

            plt.tight_layout()
            plt.suptitle('Original Distribution overlaid with SimpleImputer (Median)', fontsize=14, y=1.02)
            st.pyplot(fig)
            plt.close()
#-------------------------------------------------------------------KNN imputer plot -------------------------------------------------------------
            with col2:
                data_centers_hybrid_imputed = df.copy()

                median_columns = ['floor_space_sqft_total', 'power_capacity_MW_total']
                median_imputer = SimpleImputer(strategy='median')
                for col in median_columns:
                    if data_centers_hybrid_imputed[col].isnull().any():
                        data_centers_hybrid_imputed[col] = median_imputer.fit_transform(
                            data_centers_hybrid_imputed[[col]]
                        )

                knn_columns = ['average_renewable_energy_usage_percent', 
                            'internet_penetration_percent',
                            'growth_rate_of_data_centers_percent_per_year']

                knn_data = data_centers_hybrid_imputed[knn_columns + complete_numerical_cols]
                data_centers_without_missing = knn_data.dropna()

                if len(data_centers_without_missing) > 0:
                    scaler = StandardScaler()
                    
                    complete_rows = knn_data.dropna()
                    scaler.fit(complete_rows)
                    
                    knn_data_scaled = pd.DataFrame(
                        scaler.transform(knn_data),
                        columns=knn_data.columns,
                        index=knn_data.index
                    )
                    
                    knn_imputer = KNNImputer(n_neighbors=5)
                    knn_imputed_scaled = knn_imputer.fit_transform(knn_data_scaled)
                    
                    knn_imputed = pd.DataFrame(
                        scaler.inverse_transform(knn_imputed_scaled),
                        columns=knn_data.columns,
                        index=knn_data.index
                    )
                    
                    for col in knn_columns:
                        data_centers_hybrid_imputed[col] = knn_imputed[col]
                else:
                    print("Not enough complete rows for KNN, using median instead")
                    for col in knn_columns:
                        if data_centers_hybrid_imputed[col].isnull().any():
                            data_centers_hybrid_imputed[col] = median_imputer.fit_transform(
                                data_centers_hybrid_imputed[[col]]
                            )

                fig, axes = plt.subplots(3, 2, figsize=(10, 8))
                axes = axes.flatten()

                for idx, col in enumerate(numerical_cols_with_missing):
                    ax = axes[idx]
                    
                    original_data = df[col].dropna()
                    ax.hist(original_data, bins=30, alpha=0.5, 
                            label=f'Original (n={len(original_data)})', 
                            color='blue', edgecolor='black')
                    
                    imputed_complete = data_centers_hybrid_imputed[col]
                    ax.hist(imputed_complete, bins=30, alpha=0.5, 
                            label=f'After Hybrid Imputation (n={len(imputed_complete)})', 
                            color='green', edgecolor='black')
                    
                    if col in median_columns:
                        method = "MEDIAN"
                    else:
                        method = "KNN"
                    
                    ax.set_title(f'{col}\n(Method: {method})', fontsize=10)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                fig.delaxes(axes[5])
                plt.tight_layout()
                plt.suptitle('Hybrid Imputation Strategy: Median + KNN (K=5)', 
                            fontsize=14, y=1.02)
                st.pyplot(fig)
                plt.close()
    with st.expander("### ESG Performance Dataset Overview", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                esg_data = pd.read_csv('company_esg_financial_dataset.csv')
                columns_to_drop = [
                    'CompanyID', 
                    'CompanyName', 
                    'Revenue', 
                    'ProfitMargin', 
                    'MarketCap', 
                    'GrowthRate',
                    'ESG_Social', 
                    'ESG_Governance'
                ]

                esg_data = esg_data.drop(columns=columns_to_drop)
                esg_numeric = esg_data.select_dtypes(include=[np.number])

                if esg_numeric.shape[1] >= 2:
                    g = sns.pairplot(esg_numeric, diag_kind='kde')
                    g.fig.suptitle("Pair plot (before log-scaling)", fontsize=16, y=1.02)
                    st.pyplot(g.fig)
                    plt.close(g.fig)
                else:
                    st.write("Not enough numeric columns to display a pairplot. Numeric columns found:", list(esg_numeric.columns))
            with col2:
                esg_tech = esg_data[esg_data['Industry'] == 'Technology'].copy()

                columns_to_transform = ['CarbonEmissions', 'WaterUsage', 'EnergyConsumption']
                for col in columns_to_transform:
                    if col in esg_tech.columns and esg_tech[col].dtype.kind in 'bifc':
                        orig_skew = esg_tech[col].skew()
                        esg_tech[col] = np.log1p(esg_tech[col])
                        new_skew = esg_tech[col].skew()
                        # revert if log made skew worse
                        if abs(new_skew) >= abs(orig_skew):
                            esg_tech[col] = esg_data[esg_data['Industry'] == 'Technology'][col]

                esg_tech_numeric = esg_tech.select_dtypes(include=[np.number])

                if esg_tech_numeric.shape[1] >= 2:
                    g2 = sns.pairplot(esg_tech_numeric, diag_kind='kde')
                    g2.fig.suptitle("Pair plot (Technology - post log-scaling)", fontsize=16, y=1.02)
                    st.pyplot(g2.fig)
                    plt.close(g2.fig)
                else:
                    st.write("Not enough numeric columns in Technology subset:", list(esg_tech_numeric.columns))

with tab3:
    st.write("Plot 1")
    df = pd.read_csv('merged_datacenters_esg_timeseries.csv')
    plot1_data = df.groupby(['region', 'Year']).agg({
        'CarbonEmissions': 'mean',
        'WaterUsage': 'mean',
        'EnergyConsumption': 'mean',
        'ESG_Environmental': 'mean'
    }).reset_index()

    fig1 = px.line(
        plot1_data,
        x='Year',
        y='CarbonEmissions',
        color='region',
        markers=True,
        title='Carbon Emissions Over Time by Region (2015-2025)',
        labels={
            'CarbonEmissions': 'Carbon Emissions (log-transformed)',
            'Year': 'Year',
            'region': 'Region'
        },
        template='plotly_white'
    )

    fig1.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(title='Region', orientation='v')
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.write("Plot 2")
    plot2_data = df[df['Year'] == 2025]

    fig2 = px.scatter(plot2_data,
                    x='power_capacity_MW_total',
                    y='CarbonEmissions',
                    size='total_data_centers',
                    color='region',
                    hover_data=['country', 'average_renewable_energy_usage_percent', 
                                'ESG_Environmental'],
                    title='Data Center Power Capacity vs Carbon Emissions (2025)',
                    labels={
                        'power_capacity_MW_total': 'Power Capacity (MW, log-transformed)',
                        'CarbonEmissions': 'Carbon Emissions (log-transformed)',
                        'total_data_centers': 'Total Data Centers',
                        'region': 'Region'
                    },
                    template='plotly_white',
                    opacity=0.7)

    fig2.update_layout(height=600)
    st.plotly_chart(fig2, use_container_width=True)

with tab1:
    st.markdown("## Datasets Overview")
    st.markdown("Two complementary datasets procured from Kaggle to explore the intersection of data center infrastructure and environmental performance.")
    
    # Dataset 1
    st.markdown("### 1. Data Center Infrastructure Dataset")
    st.markdown("""
    Country-level insights into global data center infrastructure (as of 2025):
    
    **Key Features:**
    - Data center counts (total, hyperscale, colocation).
    - Power capacity in megawatts (MW).
    - Average renewable energy usage (%).
    - Average growth rates, cooling technologies, and tier distribution.
    """)
    st.page_link("https://www.kaggle.com/datasets/rockyt07/data-center-dataset/data", 
                 label="Data Center Dataset", icon="🏢")
    
    st.markdown("---")
    
    # Dataset 2
    st.markdown("### 2. ESG Performance Dataset")
    st.markdown("""
    Simulated financial and ESG metrics for 1,000 global companies (2015-2025), focused primarily on ESG measures:
    Data was subsetted to only include the Tech industry for relevance to data centers.
    
    **Key Features:**
    - 11,000 rows across 9 industries and 7 regions.
    - Financial metrics: revenue, profit margins, market capitalization.
    - ESG indicators: carbon emissions, resource usage, water usage.
    
    """)
    st.page_link("https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset", 
                 label="ESG Dataset", icon="📈")
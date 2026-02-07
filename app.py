import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stMetric {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1f77b4;
            text-align: center;
            font-size: 3em;
            margin-bottom: 10px;
        }
        h2 {
            color: #ff7f0e;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 1.1em;
            font-weight: bold;
            color: #1f77b4;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            border-bottom: 3px solid #ff7f0e;
            color: #ff7f0e;
        }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('lung_cancer.csv')
    # Clean data
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    return df.dropna(subset=['age'])

df = load_data()

# Set color palettes
color_palette_1 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
color_palette_2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("## 🫁 Lung Cancer Analysis Dashboard")
    st.markdown("<p style='text-align: center; color: #666;'>Comprehensive Multi-Dimensional Data Exploration</p>", 
                unsafe_allow_html=True)

st.markdown("---")

# Sidebar Configuration
st.sidebar.title("⚙️ Dashboard Configuration")
st.sidebar.markdown("---")

# Filter options
st.sidebar.subheader("🔍 Data Filters")
selected_gender = st.sidebar.multiselect(
    "Select Gender:", 
    df['gender'].unique(), 
    default=df['gender'].unique(),
    key="gender_filter"
)
selected_smoker = st.sidebar.multiselect(
    "Select Smoking Status:", 
    df['smoker'].unique(), 
    default=df['smoker'].unique(),
    key="smoker_filter"
)
selected_race = st.sidebar.multiselect(
    "Select Race:",
    df['race'].unique(),
    default=df['race'].unique(),
    key="race_filter"
)

# Apply filters
filtered_df = df[
    (df['gender'].isin(selected_gender)) & 
    (df['smoker'].isin(selected_smoker)) &
    (df['race'].isin(selected_race))
]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Records After Filtering: **{len(filtered_df)}** out of {len(df)}")

# Show data preview
if st.sidebar.checkbox("📋 Show Data Preview", value=False):
    st.sidebar.write(filtered_df.head(5))

st.markdown("---")

# Check if filtered data is empty
if len(filtered_df) == 0:
    st.error("❌ No data matches your selected filters. Please adjust your filters and try again.")
else:
    # Key Metrics Section
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "👥 Total Records",
            f"{len(filtered_df):,}",
            f"{len(filtered_df) - len(df)} from original"
        )

    with col2:
        male_count = len(filtered_df[filtered_df['gender'] == 'Male'])
        st.metric(
            "👨 Male",
            f"{male_count:,}",
            f"{(male_count/len(filtered_df)*100):.1f}%"
        )

    with col3:
        female_count = len(filtered_df[filtered_df['gender'] == 'Female'])
        st.metric(
            "👩 Female",
            f"{female_count:,}",
            f"{(female_count/len(filtered_df)*100):.1f}%"
        )

    with col4:
        avg_age = filtered_df['age'].mean()
        st.metric(
            "📅 Avg Age",
            f"{avg_age:.1f}",
            f"Range: {filtered_df['age'].min():.0f}-{filtered_df['age'].max():.0f}"
        )

    with col5:
        smoker_count = len(filtered_df[filtered_df['smoker'] == 'Current'])
        st.metric(
            "🚬 Current Smokers",
            f"{smoker_count:,}",
            f"{(smoker_count/len(filtered_df)*100):.1f}%"
        )

    st.markdown("---")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Demographics",
        "🚬 Smoking Analysis",
        "📈 Statistical Analysis",
        "🔥 Correlations & Heatmaps",
        "🎯 Advanced Insights"
    ])

    # TAB 1: Demographics
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Age Distribution (Histogram)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(filtered_df['age'], bins=40, color='#1f77b4', edgecolor='#0d3b66', 
                    linewidth=1.5, alpha=0.85)
            ax.set_xlabel("Age (years)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Age Distribution (Density Plot)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(filtered_df['age'], bins=40, density=True, alpha=0.7, 
                    color='#ff7f0e', edgecolor='#0d3b66', linewidth=1.5)
            ax.set_xlabel("Age (years)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Density", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Gender Distribution (Pie Chart)")
            gender_counts = filtered_df['gender'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#FF6B6B', '#4ECDC4']
            wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, 
                                               autopct='%1.1f%%', colors=colors, startangle=90,
                                               textprops={'fontsize': 11, 'fontweight': 'bold'},
                                               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Race Distribution (Bar Chart)")
            race_counts = filtered_df['race'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.viridis(np.linspace(0, 1, len(race_counts)))
            bars = ax.bar(race_counts.index, race_counts.values, color=colors, 
                          edgecolor='#333', linewidth=1.5, alpha=0.85)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            ax.set_xlabel("Race", fontsize=11, fontweight='bold')
            ax.set_ylabel("Count", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Age by Gender (Box Plot)")
            fig, ax = plt.subplots(figsize=(10, 5))
            gender_list = sorted(filtered_df['gender'].unique())
            data_to_plot = [filtered_df[filtered_df['gender'] == g]['age'].values for g in gender_list]
            bp = ax.boxplot(data_to_plot, labels=gender_list, patch_artist=True,
                           widths=0.6, showmeans=True)
            colors = ['#FF6B6B', '#4ECDC4']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel("Age (years)", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Age by Race (Violin Plot)")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(data=filtered_df, x='race', y='age', ax=ax, palette='Set2')
            ax.set_xlabel("Race", fontsize=11, fontweight='bold')
            ax.set_ylabel("Age (years)", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)

    # TAB 2: Smoking Analysis
    with tab2:
        col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Smoking Status Distribution")
        smoker_counts = filtered_df['smoker'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#FFD93D', '#FF6B6B', '#95E1D3']
        wedges, texts, autotexts = ax.pie(smoker_counts.values, labels=smoker_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'},
                                           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Smoking Status by Gender")
        cross_tab = pd.crosstab(filtered_df['gender'], filtered_df['smoker'])
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(cross_tab.index))
        width = 0.25
        colors = ['#FFD93D', '#FF6B6B', '#95E1D3']
        for i, status in enumerate(cross_tab.columns):
            ax.bar(x + i*width, cross_tab[status].values, width, label=status, 
                  color=colors[i], alpha=0.8, edgecolor='black')
        ax.set_xlabel("Gender", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=11, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(cross_tab.index)
        ax.legend(title="Smoking Status")
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Age Distribution by Smoking Status")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=filtered_df, x='smoker', y='age', ax=ax, 
                   palette=['#FFD93D', '#FF6B6B', '#95E1D3'])
        sns.stripplot(data=filtered_df, x='smoker', y='age', ax=ax, 
                     color='black', alpha=0.3, size=3)
        ax.set_xlabel("Smoking Status", fontsize=11, fontweight='bold')
        ax.set_ylabel("Age (years)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Smoking Status by Race")
        fig, ax = plt.subplots(figsize=(10, 5))
        cross_tab = pd.crosstab(filtered_df['race'], filtered_df['smoker'])
        cross_tab.plot(kind='bar', ax=ax, color=['#FFD93D', '#FF6B6B', '#95E1D3'], 
                      edgecolor='black', alpha=0.8, width=0.8)
        ax.set_title("Smoking Status by Race", fontsize=13, fontweight='bold')
        ax.set_xlabel("Race", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=11, fontweight='bold')
        ax.legend(title="Smoking Status", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)

    # TAB 3: Statistical Analysis
    with tab3:
        col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Age Distribution by Gender & Smoking")
        fig, ax = plt.subplots(figsize=(10, 5))
        for gender in filtered_df['gender'].unique():
            for smoker in filtered_df['smoker'].unique():
                data = filtered_df[(filtered_df['gender'] == gender) & (filtered_df['smoker'] == smoker)]['age']
                ax.hist(data, bins=25, alpha=0.4, label=f"{gender} - {smoker}")
        ax.set_xlabel("Age (years)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Age Statistics Summary")
        stats_df = filtered_df.groupby('gender')['age'].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("### Smoking Status Statistics")
        smoker_stats = filtered_df['smoker'].value_counts()
        for status, count in smoker_stats.items():
            percentage = (count / len(filtered_df) * 100)
            st.write(f"**{status}:** {count:,} ({percentage:.1f}%)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Scatter: Age vs Gender")
        sample_df = filtered_df.sample(min(1000, len(filtered_df)))
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_map = {'Current': '#E74C3C', 'Former': '#3498DB', 'Never': '#2ECC71'}
        for smoker_status in sample_df['smoker'].unique():
            mask = sample_df['smoker'] == smoker_status
            ax.scatter(sample_df[mask]['gender'], sample_df[mask]['age'], 
                      label=smoker_status, alpha=0.6, s=sample_df[mask]['age']*2,
                      color=colors_map.get(smoker_status, '#95A5A6'))
        ax.set_xlabel("Gender", fontsize=11, fontweight='bold')
        ax.set_ylabel("Age (years)", fontsize=11, fontweight='bold')
        ax.legend(title="Smoking Status")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 2D Distribution: Gender vs Race")
        fig, ax = plt.subplots(figsize=(10, 5))
        cross_data = pd.crosstab(filtered_df['gender'], filtered_df['race'])
        im = ax.imshow(cross_data.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(cross_data.columns)))
        ax.set_yticks(range(len(cross_data.index)))
        ax.set_xticklabels(cross_data.columns, rotation=45, ha='right')
        ax.set_yticklabels(cross_data.index)
        for i in range(len(cross_data.index)):
            for j in range(len(cross_data.columns)):
                ax.text(j, i, str(cross_data.values[i, j]), ha='center', va='center', color='black', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Count')
        st.pyplot(fig, use_container_width=True)

    # TAB 4: Correlations & Heatmaps
    with tab4:
        col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Heatmap: Gender vs Race")
        fig, ax = plt.subplots(figsize=(10, 5))
        heatmap_data = pd.crosstab(filtered_df['gender'], filtered_df['race'])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='RdYlGn', ax=ax, 
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        ax.set_title("Gender vs Race Heatmap", fontsize=12, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Heatmap: Smoking Status vs Race")
        fig, ax = plt.subplots(figsize=(10, 5))
        heatmap_data = pd.crosstab(filtered_df['smoker'], filtered_df['race'])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        ax.set_title("Smoking Status vs Race Heatmap", fontsize=12, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Heatmap: Gender vs Smoking Status")
        fig, ax = plt.subplots(figsize=(10, 5))
        heatmap_data = pd.crosstab(filtered_df['gender'], filtered_df['smoker'])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Purples', ax=ax,
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        ax.set_title("Gender vs Smoking Status Heatmap", fontsize=12, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Age Range Distribution Heatmap")
        age_bins = pd.cut(filtered_df['age'], bins=6)
        fig, ax = plt.subplots(figsize=(10, 5))
        heatmap_data = pd.crosstab(age_bins, filtered_df['gender'])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        ax.set_title("Age Range vs Gender Heatmap", fontsize=12, fontweight='bold')
        ax.set_xlabel("Gender", fontsize=11, fontweight='bold')
        st.pyplot(fig, use_container_width=True)

    # TAB 5: Advanced Insights
    with tab5:
        col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Age Distribution Curves by Gender")
        fig, ax = plt.subplots(figsize=(10, 5))
        for gender in filtered_df['gender'].unique():
            data = filtered_df[filtered_df['gender'] == gender]['age']
            ax.hist(data, bins=30, density=True, alpha=0.5, label=gender)
        ax.set_xlabel("Age (years)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Density", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Cumulative Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#FF6B6B', '#4ECDC4']
        for i, gender in enumerate(sorted(filtered_df['gender'].unique())):
            data = filtered_df[filtered_df['gender'] == gender]['age'].sort_values()
            cumulative = np.arange(1, len(data) + 1) / len(data) * 100
            ax.plot(data.values, cumulative, label=gender, linewidth=2.5, 
                   color=colors[i], marker='o', markersize=3, alpha=0.7)
        ax.set_xlabel("Age (years)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Cumulative Percentage (%)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Gender Distribution by Age Groups")
        age_groups = pd.cut(filtered_df['age'], bins=[0, 30, 40, 50, 60, 70, 80, 100],
                           labels=['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '>80'])
        fig, ax = plt.subplots(figsize=(10, 5))
        cross_tab = pd.crosstab(age_groups, filtered_df['gender'])
        cross_tab.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], 
                      edgecolor='black', alpha=0.8, width=0.7)
        ax.set_title("Gender Distribution by Age Groups", fontsize=12, fontweight='bold')
        ax.set_xlabel("Age Group", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=11, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Smoking Status by Age Groups")
        fig, ax = plt.subplots(figsize=(10, 5))
        cross_tab = pd.crosstab(age_groups, filtered_df['smoker'])
        cross_tab.plot(kind='area', ax=ax, alpha=0.7, color=['#FFD93D', '#FF6B6B', '#95E1D3'])
        ax.set_title("Smoking Status Distribution by Age Groups", fontsize=12, fontweight='bold')
        ax.set_xlabel("Age Group", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=11, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Summary insights
    st.markdown("### 💡 Key Insights & Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Age Analysis**
        - Mean: {filtered_df['age'].mean():.1f} years
        - Median: {filtered_df['age'].median():.1f} years
        - Std Dev: {filtered_df['age'].std():.1f} years
        - Range: {filtered_df['age'].min():.0f}-{filtered_df['age'].max():.0f} years
        """)
    
    with col2:
        male_pct = len(filtered_df[filtered_df['gender'] == 'Male']) / len(filtered_df) * 100
        st.success(f"""
        **Gender Distribution**
        - Male: {male_pct:.1f}%
        - Female: {100-male_pct:.1f}%
        """)
    
    with col3:
        current_smoker_pct = len(filtered_df[filtered_df['smoker'] == 'Current']) / len(filtered_df) * 100
        st.warning(f"""
        **Smoking Status**
        - Current: {current_smoker_pct:.1f}%
        - Former: {len(filtered_df[filtered_df['smoker'] == 'Former']) / len(filtered_df) * 100:.1f}%
        - Never: {len(filtered_df[filtered_df['smoker'] == 'Never']) / len(filtered_df) * 100:.1f}%
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em;'>
    🔬 Lung Cancer Analysis Dashboard | Data Visualization Report<br>
    <small>Generated with Streamlit, Plotly, Matplotlib & Seaborn</small>
</div>
""", unsafe_allow_html=True)

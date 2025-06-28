from matplotlib import cm
from matplotlib.colors import to_rgba
import streamlit as st
import pandas as pd
import numpy as np          # <-- ✅ Add this line
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk        # <-- ✅ Required for custom map
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib

# ------------------------------- Page Config
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------- Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv("data/student_dataset.csv")

def load_model():
    return joblib.load("models/best_random_forest_model.pkl")

df = load_data()
model = load_model()

# ------------------------------- Title
st.title("📊 Student Academic Performance Dashboard")

# ------------------------------- Sidebar: Filter and Predict
with st.sidebar:
    # ---------- Extract Options
    grade_options = sorted(df['GradeID'].dropna().unique())
    topic_options = sorted(df['Topic'].dropna().unique())
    nationality_options = sorted(df['NationalITy'].dropna().unique())
    gender_options = sorted(df['gender'].dropna().unique())

    # ---------- Filter Panel
    with st.expander("🔍 Filter Students", expanded=True):
        selected_grade = st.multiselect("Select Grade", options=grade_options, default=[])
        selected_topic = st.multiselect("Select Subject", options=topic_options, default=[])
        selected_nationality = st.multiselect("Select Nationality", options=nationality_options, default=[])
        selected_gender = st.multiselect("Select Gender", options=gender_options, default=[])

        # Reset filters to all if nothing selected
        if not selected_grade:
            selected_grade = grade_options
        if not selected_topic:
            selected_topic = topic_options
        if not selected_nationality:
            selected_nationality = nationality_options
        if not selected_gender:
            selected_gender = gender_options

        # Apply filter
        filtered_df = df[
            (df['GradeID'].isin(selected_grade)) &
            (df['Topic'].isin(selected_topic)) &
            (df['NationalITy'].isin(selected_nationality)) &
            (df['gender'].isin(selected_gender))
        ]

    st.markdown("---")

    # ---------- Predict Panel
    with st.expander("🎯 Predict Student Performance", expanded=True):
        features_to_include = ['VisITedResources', 'raisedhands', 'AnnouncementsView', 'StudentAbsenceDays', 'Discussion']

        with st.form("predict_form_sidebar"):
            visited = st.slider("Visited Resources", 0, 100, 30, key="visited_sidebar")
            hands = st.slider("Raised Hands", 0, 100, 20, key="hands_sidebar")
            announcements = st.slider("Announcements Viewed", 0, 100, 15, key="announcements_sidebar")
            discussion = st.slider("Discussion Participation", 0, 100, 10, key="discussion_sidebar")
            absence = st.radio("Absence", ["Under-7", "Above-7"], key="absence_sidebar")

            submitted_sidebar = st.form_submit_button("🚀 Predict")

        if submitted_sidebar:
            input_df = pd.DataFrame([{
                'VisITedResources': visited,
                'raisedhands': hands,
                'AnnouncementsView': announcements,
                'StudentAbsenceDays': 1 if absence == 'Above-7' else 0,
                'Discussion': discussion
            }])[features_to_include]

            pred_class = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            confidence = np.max(pred_proba) * 100  # as percentage

            label_map = {
                0: ("Low", "#d9534f"),
                1: ("Medium", "#f0ad4e"),
                2: ("High", "#5cb85c")
            }
            label, color = label_map.get(pred_class, ("Unknown", "gray"))

            st.markdown(f"""
            <div style='
                padding: 0.75rem;
                border-radius: 6px;
                background-color: {color};
                color: white;
                font-weight: bold;
                text-align: center;
                font-size: 16px;'>
                🎯 Predicted Class: {label}<br>
                🔒 Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

            st.caption(f"Class Probabilities: {np.round(pred_proba, 2)}")

            # Risk logic
            risk_flag = (
                sum([
                    visited < df['VisITedResources'].mean(),
                    hands < df['raisedhands'].mean(),
                    announcements < df['AnnouncementsView'].mean(),
                    discussion < df['Discussion'].mean()
                ]) >= 2
                and absence == 'Above-7'
                and label == 'Low'
            )

            if risk_flag:
                st.error("⚠️ At-Risk Student")
            else:
                st.success("✅ Not At-Risk")


# ------------------------------- 🎓 Metric Cards Section -------------------------------

# Calculate metrics
total_students = len(filtered_df)
total_male = (filtered_df['gender'] == 'M').sum()
total_female = (filtered_df['gender'] == 'F').sum()
at_risk_pct = round((filtered_df['Class'] == 'L').mean() * 100, 2)
most_challenging_subject = filtered_df[filtered_df['Class'] == 'L']['Topic'].value_counts().idxmax()

# Minimal, professional style
st.markdown("""
    <style>
        .metric-box {
            background-color: #f5f7fa;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            text-align: center;
            height: 140px;
            color: #333333;
            font-family: "Segoe UI", sans-serif;
        }
        .metric-label {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #555555;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 6px;
            color: #222222;
        }
        .metric-sub {
            font-size: 13px;
            color: #777777;
        }
    </style>
""", unsafe_allow_html=True)

# Layout with 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">🎯 Total Students</div>
            <div class="metric-value">{total_students}</div>
            <div class="metric-sub">M: {total_male} | F: {total_female}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">🚨 At-Risk Students</div>
            <div class="metric-value">{at_risk_pct}%</div>
            <div class="metric-sub">Based on 'L' in Class</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">📘 Most Challenging Subject</div>
            <div class="metric-value">{most_challenging_subject}</div>
            <div class="metric-sub">Among At-Risk Students</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

# ------------------------------- Charts Row 1
with st.container():
    # Add spacing column in between
    row1_col1, spacer, row1_col2 = st.columns([1, 0.25, 1.4])

    # --- Left: Performance Level Distribution ---
    with row1_col1:
        st.markdown("##### 📈 Performance Level Distribution")

        # Map and prepare class labels
        class_label_map = {'H': '<b>High</b>', 'M': '<b>Medium</b>', 'L': '<b>Low</b>'}
        class_series = filtered_df['Class'].map(class_label_map)
        class_counts = class_series.value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']  # ✅ Ensure unique column names

        # palette
        color_map = {
            '<b>High</b>': '#1f77b4',   
            '<b>Medium</b>': '#eca532', 
            '<b>Low</b>': '#8B0000'     
        }

        fig_donut = px.pie(
            class_counts,
            names='Class',
            values='Count',
            hole=0.4,
            color='Class',
            color_discrete_map=color_map
        )

        fig_donut.update_traces(
            textposition='inside',
            textinfo='label+percent+value',
            textfont=dict(size=13),
            marker=dict(line=dict(color='#FFFFFF', width=1))
        )

        fig_donut.update_layout(
            showlegend=True,
            legend_title_text='Performance Level',
            font=dict(family="Segoe UI", size=12),
            margin=dict(t=20, b=0, l=0, r=0)
        )

        st.plotly_chart(fig_donut, use_container_width=True)

    # --- Right: Student Nationality Map ---
    with row1_col2:
        st.markdown("##### 🗺️ Student Nationality Map")

        country_name_map = {
            'USA': 'United States of America',
            'Tunis': 'Tunisia',
            'Lybia': 'Libya',
            'Venzuela': 'Venezuela',
            'Kuwait': 'Kuwait',
            'Lebanon': 'Lebanon',
            'Egypt': 'Egypt',
            'Saudi Arabia': 'Saudi Arabia',
            'Jordan': 'Jordan',
            'Iran': 'Iran',
            'Morocco': 'Morocco',
            'Syria': 'Syria',
            'Palestine': 'Palestine',
            'Iraq': 'Iraq'
        }

        filtered_df['NationalITy'] = filtered_df['NationalITy'].replace(country_name_map)

        nationality_counts = (
            filtered_df.groupby('NationalITy')['ID']
            .count()
            .reset_index()
            .rename(columns={'NationalITy': 'country', 'ID': 'students'})
        )
        total_students = nationality_counts['students'].sum()
        nationality_counts['percentage'] = (nationality_counts['students'] / total_students * 100).round(2)

        url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
        response = requests.get(url)
        world_geo = response.json()

        country_coords = {
            'United States of America': {"lat": 37.1, "lon": -95.7},
            'Tunisia': {"lat": 34.0, "lon": 9.0},
            'Libya': {"lat": 26.0, "lon": 17.0},
            'Venezuela': {"lat": 6.4, "lon": -66.6},
            'Kuwait': {"lat": 29.3, "lon": 47.5},
            'Lebanon': {"lat": 33.8, "lon": 35.8},
            'Egypt': {"lat": 26.8, "lon": 30.8},
            'Saudi Arabia': {"lat": 24.0, "lon": 45.0},
            'Jordan': {"lat": 31.0, "lon": 36.0},
            'Iran': {"lat": 32.0, "lon": 53.0},
            'Morocco': {"lat": 31.8, "lon": -6.0},
            'Syria': {"lat": 35.0, "lon": 38.5},
            'Palestine': {"lat": 31.9, "lon": 35.2},
            'Iraq': {"lat": 33.0, "lon": 44.0}
        }

        nationality_counts['lat'] = nationality_counts['country'].map(lambda x: country_coords.get(x, {}).get('lat'))
        nationality_counts['lon'] = nationality_counts['country'].map(lambda x: country_coords.get(x, {}).get('lon'))

        fig_map = px.choropleth(
            nationality_counts,
            geojson=world_geo,
            locations='country',
            featureidkey='properties.name',
            color='students',
            color_continuous_scale='YlOrRd',
            hover_name='country',
            hover_data={'students': True, 'percentage': True},
            projection='natural earth',
            height=450
        )

        # ✅ Display country names only when zoomed (≤ 3 countries shown)
        if len(nationality_counts) <= 3:
            fig_map.add_trace(
                go.Scattergeo(
                    lon=nationality_counts['lon'],
                    lat=nationality_counts['lat'],
                    text=nationality_counts['country'],
                    mode='text',
                    textfont=dict(size=10, color='black'),
                    showlegend=False
                )
            )

        fig_map.update_geos(
            showcountries=True,
            countrycolor="gray",
            showcoastlines=True,
            coastlinecolor="white",
            showland=True,
            landcolor="whitesmoke",
            showocean=True,
            oceancolor="lightblue",
            projection_type='natural earth'
        )

        if len(nationality_counts) == 1:
            single_country = nationality_counts['country'].iloc[0]
            center = country_coords.get(single_country, {"lat": 20.0, "lon": 0.0})
            fig_map.update_geos(center=center, projection_scale=3)

        fig_map.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_colorbar=dict(title='Students'))

        # ✅ Show map
        st.plotly_chart(fig_map, use_container_width=True)

        # ✅ Below-map horizontal annotation for top 3 countries
        top3 = nationality_counts.sort_values(by='students', ascending=False).head(3)
        st.markdown(
            f"<div style='margin-top: 5px; background-color: #f9f9f9; padding: 10px; border-radius: 8px; text-align:center;'>"
            + " &nbsp; | &nbsp; ".join([
                f"<b>{row['country']}</b>: {row['students']} students ({row['percentage']}%)"
                for _, row in top3.iterrows()
            ])
            + "</div>",
            unsafe_allow_html=True
        )

st.markdown("""
<div style='
    margin-top: 40px;
    margin-bottom: 30px;
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    text-align: center;
    color: #333333;
    font-family: "Segoe UI", sans-serif;
    font-size: 26px;
    font-weight: 700;
'>
📊 In-Depth Student Performance Analytics
</div>
""", unsafe_allow_html=True)


# Define consistent color mapping
class_color_map = {'H': '#1f77b4', 'M': "#eca532", 'L': "#8B0000"}
class_order = ['H', 'M', 'L']  # For sorting legends

# Row 1: Gender vs Performance (Stacked Bar) and At-Risk Heatmap
with st.container():
    row1_col1, spacer1, row1_col2 = st.columns([1, 0.2, 1])

    # --- Chart 1: Stacked Bar - Gender vs Performance ---
    with row1_col1:
        st.markdown("##### 👩‍🎓 Performance Distribution by Gender")
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        gender_perf = filtered_df.groupby(['gender', 'Class']).size().reset_index(name='Count')
        gender_perf['Class'] = pd.Categorical(gender_perf['Class'], categories=class_order, ordered=True)
        gender_perf = gender_perf.sort_values(by='Class')

        fig_gender = px.bar(
            gender_perf, x='gender', y='Count', color='Class', barmode='stack',
            color_discrete_map=class_color_map, category_orders={'Class': class_order}
        )
        fig_gender.update_layout(xaxis_title=None, yaxis_title=None)  # ✅ Remove axis titles
        st.plotly_chart(fig_gender, use_container_width=True)


    # --- Chart 2: Heatmap - At-Risk by Grade & Section ---
    with row1_col2:
        st.markdown("##### 🔥 At-Risk Heatmap by Grade and Section")
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        filtered_df['AtRisk'] = filtered_df['Class'] == 'L'
        risk_heatmap = pd.crosstab(
            filtered_df['GradeID'],
            filtered_df['SectionID'],
            filtered_df['AtRisk'],
            aggfunc='sum'
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(risk_heatmap, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
        st.pyplot(fig)

st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

# Row 2: Line Chart (Engagement) and Grouped Bar (Subject)
with st.container():
    row2_col1, spacer2, row2_col2 = st.columns([1, 0.2, 1])

    # --- Chart 3: Line Chart - Engagement Behavior by Performance ---
    with row2_col1:
        st.markdown("##### 📈 Average Engagement Behavior by Performance")
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        engagement_cols = ['VisITedResources', 'raisedhands', 'AnnouncementsView', 'Discussion']
        engagement_avg = filtered_df.groupby('Class')[engagement_cols].mean().reset_index()
        engagement_avg['Class'] = pd.Categorical(engagement_avg['Class'], categories=class_order, ordered=True)
        engagement_avg_melted = engagement_avg.melt(id_vars='Class', var_name='Behavior', value_name='AvgScore')

        fig_line = px.line(
            engagement_avg_melted,
            x='Behavior',
            y='AvgScore',
            color='Class',
            markers=True,
            color_discrete_map=class_color_map,
            category_orders={'Class': class_order}
        )
        fig_line.update_traces(marker=dict(size=8), line=dict(width=3))
        fig_line.update_layout(
            title=None,  # ✅ Remove chart title
            xaxis_title=None,  # ✅ Remove x-axis title
            yaxis_title=None   # ✅ Remove y-axis title
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # --- Chart 4: Grouped Bar - Subject-wise Performance ---
    with row2_col2:
        st.markdown("##### 📚 Subject-wise Performance Breakdown")
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        subject_perf = filtered_df.groupby(['Topic', 'Class']).size().reset_index(name='Count')
        subject_perf['Class'] = pd.Categorical(subject_perf['Class'], categories=class_order, ordered=True)
        subject_perf = subject_perf.sort_values(by=['Topic', 'Class'])

        fig_subject = px.bar(
            subject_perf,
            x='Topic',
            y='Count',
            color='Class',
            barmode='group',
            color_discrete_map=class_color_map,
            category_orders={'Class': class_order}
        )
        fig_subject.update_layout(
            xaxis_title=None,  # ✅ Remove x-axis title
            yaxis_title=None   # ✅ Remove y-axis title
        )
        st.plotly_chart(fig_subject, use_container_width=True)


# ------------------------------- At-Risk Student Detection
st.markdown("<br><hr style='border-top: 1px solid lightgray;'/><br>", unsafe_allow_html=True)
st.subheader("🚩 At-Risk Student Records")

# Step 1: Define engagement features and calculate low engagement count
engagement_cols = ['Discussion', 'raisedhands', 'VisITedResources', 'AnnouncementsView']
low_engagement_flags = sum((filtered_df[col] < filtered_df[col].mean()).astype(int) for col in engagement_cols)

# Step 2: Apply AtRisk condition
filtered_df = filtered_df.copy()
filtered_df['AtRisk'] = (
    (low_engagement_flags >= 2) &
    (filtered_df['StudentAbsenceDays'] == 'Above-7') &
    (filtered_df['Class'] == 'L')
)

# Step 3: Display AtRisk stats
total_students = len(filtered_df)
total_at_risk = filtered_df['AtRisk'].sum()
percent_at_risk = (total_at_risk / total_students) * 100
st.info(f"🎯 {total_at_risk} out of {total_students} students are flagged as **at risk** ({percent_at_risk:.1f}%).")

# Step 4: Clean up columns for display
excluded_columns = ['PlaceofBirth', 'StageID', 'SectionID', 
                    'ParentAnsweringSurvey', 'Relation', 'ParentschoolSatisfaction']
display_df = filtered_df.drop(columns=excluded_columns, errors='ignore')

# Step 5: Display DataFrame with red highlight for at-risk students
st.dataframe(
    display_df.style.applymap(
        lambda val: 'background-color: salmon' if val is True else '',
        subset=['AtRisk']
    ),
    use_container_width=True,
    height=350
)

st.info(
    " 💡 **At-Risk ACriteria**: A student is flagged as *at risk* if they show low engagement "
    "in at least **2 of 4 behaviors** (Discussion, Raised Hands, Visited Resources, "
    "Announcements View), **and** have **more than 7 absences** and are in the **Low (L) performance class**."
)

st.markdown("<br><hr style='border-top: 1px solid lightgray;'/><br>", unsafe_allow_html=True)





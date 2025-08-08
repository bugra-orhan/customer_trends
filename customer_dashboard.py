import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Customer Dashboard", layout="wide")

# Veri Yükleme
@st.cache_data
def load_data():
    df = pd.read_csv("shopping_trends.csv")
    return df

df = load_data()

st.title("🛍️ Customer Behavior Dashboard")

# ----------------- SIDEBAR FILTERS ------------------

st.sidebar.header("🔎 Filters")

category = st.sidebar.multiselect("Category", options=df["Category"].unique(), default=df["Category"].unique())
item = st.sidebar.multiselect("Item Purchased", options=df["Item Purchased"].unique(), default=df["Item Purchased"].unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
location = st.sidebar.multiselect("Location", options=df["Location"].unique(), default=df["Location"].unique())

df_filtered = df[
    (df["Category"].isin(category)) &
    (df["Item Purchased"].isin(item)) &
    (df["Gender"].isin(gender)) &
    (df["Location"].isin(location))
]

# ------------------ KPIs ------------------

st.markdown("## 📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🛒 Total Purchases", len(df_filtered))

with col2:
    st.metric("💵 Total Revenue", f"${df_filtered['Purchase Amount (USD)'].sum():,.0f}")

with col3:
    st.metric("🧑‍🤝‍🧑 Unique Customers", df_filtered["Customer ID"].nunique())

with col4:
    avg_age = round(df_filtered["Age"].mean(), 1)
    st.metric("🎂 Avg Customer Age", avg_age)

# ------------------ VISUALIZATIONS ------------------

st.markdown("---")
st.markdown("### 📊 Subscription Status by Category")

sub_cat = df_filtered.groupby(["Category", "Subscription Status"]).size().reset_index(name="Count")

fig1 = px.bar(
    sub_cat,
    x="Category",
    y="Count",
    color="Subscription Status",
    barmode="group",
    title="Subscription Status by Category"
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 🎂 Age Distribution")

fig2 = px.histogram(df, x="Age", nbins=20, title="Customer Age Distribution")  # Kırmızı ton güzel olmuştu
fig2.update_traces(marker=dict(line=dict(width=1, color='gray')))
fig2.update_layout(bargap=0.05, plot_bgcolor='white')
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 👨‍👩‍👧‍👦 Gender Distribution")

gender_counts = df_filtered["Gender"].value_counts().reset_index()
gender_counts.columns = ["Gender", "Count"]

fig3 = px.pie(
    gender_counts,
    names="Gender",
    values="Count",
    title="Gender Distribution"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("### 🛍️ Most Purchased Items")

top_items = df_filtered["Item Purchased"].value_counts().head(10).reset_index()
top_items.columns = ["Item", "Count"]

fig4 = px.bar(
    top_items,
    x="Count",
    y="Item",
    orientation='h',
    title="Top 10 Purchased Items"
)
st.plotly_chart(fig4, use_container_width=True)

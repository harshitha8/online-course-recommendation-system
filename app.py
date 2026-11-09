import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("models/recommendation_model.pkl", "rb"))
    tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    df = pickle.load(open("models/course_dataframe.pkl", "rb"))

    tfidf_matrix = tfidf.transform(df["combined_features"])

    indices = pd.Series(
        df.index,
        index=df["course_name"]
    ).drop_duplicates()

    return model, tfidf, df, tfidf_matrix, indices

model, tfidf, df, tfidf_matrix, indices = load_model()

# ---------------- Recommendation Function ----------------
def recommend_courses(course_name):

    if course_name not in indices:
        return pd.DataFrame()

    idx = indices[course_name]

    distances, neighbors = model.kneighbors(
    tfidf_matrix[idx],
    n_neighbors=6
    )

    recommended = neighbors.flatten()[1:6]

    return df.iloc[recommended][[
        "course_name",
        "instructor",
        "difficulty_level",
        "rating",
        "course_price",
        "course_duration_hours"
    ]]

# ---------------- Sidebar ----------------
st.sidebar.title("📚 Project Information")

st.sidebar.write("""
**Project:** Online Course Recommendation System

**Recommendation Technique:** TF-IDF + Nearest Neighbors

Developed using:
- Python
- Pandas
- Scikit-Learn
- Streamlit
""")

# ---------------- Main Page ----------------
st.title("🎓 Online Course Recommendation System")

st.markdown("### Find Similar Courses Instantly")

selected_course = st.selectbox(
    "Select a Course",
    sorted(df["course_name"].drop_duplicates())
)

if st.button("🔍 Recommend Courses"):

    recommendations = recommend_courses(selected_course)

    st.success("Top 5 Recommended Courses")

    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):

        st.markdown(f"## {i}. {row['course_name']}")
        st.write(f"👨‍🏫 **Instructor:** {row['instructor']}")
        st.write(f"🎓 **Difficulty:** {row['difficulty_level']}")
        st.write(f"⭐ **Rating:** {row['rating']}")
        st.write(f"💰 **Price:** ₹{row['course_price']}")
        st.write(f"⏱️ **Duration:** {row['course_duration_hours']} Hours")

        st.markdown("---")
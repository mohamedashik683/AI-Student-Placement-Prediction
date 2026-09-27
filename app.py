import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
}

.card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #e5e7eb;
    background-color: #f8fafc;
}

.metric-title {
    font-size: 14px;
    color: #64748b;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
}

.recommendation {
    padding: 14px 18px;
    border-radius: 12px;
    background-color: #eff6ff;
    border-left: 5px solid #2563eb;
    margin-bottom: 10px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("placement_data.csv")

features = [
    "CGPA",
    "Attendance",
    "Aptitude_Score",
    "Technical_Skill_Score",
    "Communication_Score",
    "Projects",
    "Certifications",
    "Internships"
]

X = df[features]
y = df["Placement_Status"]


# =========================================================
# TRAIN MODEL
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred) * 100


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<h1>🎓 AI Student Placement Predictor</h1>

<p>
Machine Learning Based Placement Prediction &
Personalized Skill Recommendation System
</p>

</div>
""", unsafe_allow_html=True)


st.write(
    "Analyze academic performance, aptitude, technical skills "
    "and experience to estimate placement readiness."
)


# =========================================================
# MODEL STATISTICS
# =========================================================

st.markdown(
    '<div class="section-title">📊 Model Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Model", "Logistic Regression")

with c2:
    st.metric("Accuracy", f"{accuracy:.2f}%")

with c3:
    st.metric("Training Records", len(X_train))

with c4:
    st.metric("Testing Records", len(X_test))


st.caption(
    "Model performance is evaluated using the generated project dataset."
)


# =========================================================
# STUDENT PROFILE
# =========================================================

st.markdown(
    '<div class="section-title">👨‍🎓 Student Profile</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📚 Academic")

        cgpa = st.number_input(
            "CGPA",
            0.0,
            10.0,
            7.0,
            0.1
        )

        attendance = st.number_input(
            "Attendance (%)",
            0,
            100,
            75
        )

        aptitude = st.number_input(
            "Aptitude Score",
            0,
            100,
            60
        )

    with col2:

        st.markdown("### 💻 Skills")

        technical = st.number_input(
            "Technical Skill Score",
            0,
            100,
            60
        )

        communication = st.number_input(
            "Communication Score",
            0,
            100,
            60
        )

        projects = st.number_input(
            "Projects Completed",
            0,
            10,
            2
        )

    with col3:

        st.markdown("### 🏆 Experience")

        certifications = st.number_input(
            "Certifications",
            0,
            10,
            2
        )

        internships = st.number_input(
            "Internships",
            0,
            5,
            0
        )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("")

predict = st.button(
    "🔮  ANALYZE PLACEMENT",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    input_data = pd.DataFrame([[
        cgpa,
        attendance,
        aptitude,
        technical,
        communication,
        projects,
        certifications,
        internships
    ]], columns=features)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    placed_index = list(model.classes_).index("Placed")

    placement_probability = (
        probabilities[placed_index] * 100
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    r1, r2 = st.columns(2)

    with r1:

        if prediction == "Placed":

            st.success(
                "🎉 PLACEMENT PREDICTION: PLACED"
            )

        else:

            st.error(
                "⚠️ PLACEMENT PREDICTION: NOT PLACED"
            )

    with r2:

        st.metric(
            "Placement Probability",
            f"{placement_probability:.2f}%"
        )


    st.progress(
        min(int(placement_probability), 100)
    )


    # =====================================================
    # PROFILE SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Profile Summary</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("CGPA", f"{cgpa:.1f}")

    with s2:
        st.metric("Attendance", f"{attendance}%")

    with s3:
        st.metric("Projects", projects)

    with s4:
        st.metric("Internships", internships)


    # =====================================================
    # PERFORMANCE CHART
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Skill Analysis</div>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Skill": [
            "Aptitude",
            "Technical",
            "Communication"
        ],
        "Score": [
            aptitude,
            technical,
            communication
        ]
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        chart_data["Skill"],
        chart_data["Score"]
    )

    ax.set_ylim(0, 100)

    ax.set_ylabel("Score")

    ax.set_title(
        "Student Skill Performance"
    )

    st.pyplot(fig)


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">💡 Personalized Recommendations</div>',
        unsafe_allow_html=True
    )

    recommendations = []

    if technical < 70:

        recommendations.append(
            "🐍 Strengthen Python and programming skills."
        )

    if aptitude < 70:

        recommendations.append(
            "🧠 Practice aptitude and logical reasoning regularly."
        )

    if communication < 70:

        recommendations.append(
            "🗣️ Improve communication and interview skills."
        )

    if projects < 2:

        recommendations.append(
            "💻 Build more real-world AI/DS projects."
        )

    if certifications < 2:

        recommendations.append(
            "📜 Complete relevant technical certifications."
        )

    if internships == 0:

        recommendations.append(
            "🏢 Try to gain internship experience."
        )

    if attendance < 75:

        recommendations.append(
            "📚 Improve academic attendance."
        )

    if not recommendations:

        recommendations.append(
            "🔥 Strong profile! Focus on interview preparation."
        )


    for recommendation in recommendations:

        st.markdown(
            f"""
            <div class="recommendation">
            {recommendation}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    <b>AI-Based Student Placement Prediction & Skill Recommendation System</b>
    <br>
    B.Tech Artificial Intelligence & Data Science

    </div>
    """,
    unsafe_allow_html=True
)
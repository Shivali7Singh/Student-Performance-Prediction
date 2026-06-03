import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right, #4facfe, #00f2fe);
}

.main-title{
    text-align:center;
    color:white;
    font-size:50px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:20px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
}

.footer{
    text-align:center;
    color:white;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    "<h1 class='main-title'>🎓 Student Performance Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Predict Student Pass/Fail Status using Machine Learning</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("student-mat 1.csv")

# Create Pass/Fail Target
df['Result'] = df['G3'].apply(
    lambda x: 'Pass' if x >= 10 else 'Fail'
)

# Features
X = df[['studytime', 'absences', 'G1', 'G2']]
y = df['Result']

# Encode Target
le = LabelEncoder()
y = le.fit_transform(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📋 Student Details")

studytime = st.sidebar.slider(
    "Study Time",
    1, 4, 2
)

absences = st.sidebar.number_input(
    "Absences",
    min_value=0,
    value=5
)

g1 = st.sidebar.slider(
    "G1 Score",
    0, 20, 10
)

g2 = st.sidebar.slider(
    "G2 Score",
    0, 20, 10
)

st.sidebar.markdown("---")

st.sidebar.success(
    f"Model Accuracy: {round(accuracy*100,2)}%"
)

st.sidebar.info("""
### Information

🎯 Algorithm: Decision Tree Classifier

📂 Dataset: Student Performance Dataset

💻 Framework: Streamlit

🔗 Version Control: GitHub
""")

# -------------------------------------------------
# MODEL ACCURACY CARD
# -------------------------------------------------

st.markdown(f"""
<div class='card'>
<h3>📈 Model Accuracy</h3>
<h2>{round(accuracy*100,2)}%</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# STUDENT SUMMARY
# -------------------------------------------------

st.markdown("## 📊 Student Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Study Time", studytime)
c2.metric("Absences", absences)
c3.metric("G1 Score", g1)
c4.metric("G2 Score", g2)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------

if st.button("🔍 Predict Result", use_container_width=True):

    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)

    student = [[studytime, absences, g1, g2]]

    prediction = model.predict(student)

    st.markdown("---")

    if prediction[0] == 1:

        st.markdown("""
        <div style='background:#d4edda;
        padding:25px;
        border-radius:15px;
        text-align:center;'>

        <h1>✅ PASS</h1>

        <h3>The student is likely to pass.</h3>

        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    else:

        st.markdown("""
        <div style='background:#f8d7da;
        padding:25px;
        border-radius:15px;
        text-align:center;'>

        <h1>❌ FAIL</h1>

        <h3>The student may require additional support.</h3>

        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------
# DATASET SAMPLE
# -------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📄 View Dataset Sample"):
    st.dataframe(df.head(10))

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.markdown("""
<div class='footer'>
<b>Developed using Python, Scikit-Learn, Streamlit & GitHub</b>
<br>
Student Performance Prediction System
</div>
""", unsafe_allow_html=True)
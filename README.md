# 🎓 AI-Based Student Placement Prediction & Skill Recommendation System

## 📌 About the Project

This project is a Machine Learning based web application that predicts a student's placement status based on academic performance, aptitude, technical skills, communication skills, projects, certifications, and internship experience.

The system also provides personalized skill recommendations to help students improve their placement readiness.

## 🎯 Objectives

- Predict student placement status using Machine Learning.
- Estimate placement probability.
- Analyze student skills and academic performance.
- Provide personalized skill improvement recommendations.
- Build an interactive web application using Streamlit.

## 🧠 Machine Learning

Two classification algorithms were evaluated:

- Logistic Regression
- Random Forest

### Best Performing Model

**Logistic Regression**

| Metric | Score |
|---|---:|
| Accuracy | 98.00% |
| Precision | 100.00% |
| Recall | 95.65% |
| F1 Score | 97.78% |

> Note: These results were obtained using the generated project dataset and are intended for educational/project demonstration purposes.

## 📊 Input Features

The model uses:

- CGPA
- Attendance
- Aptitude Score
- Technical Skill Score
- Communication Score
- Number of Projects
- Certifications
- Internships

## ✨ Key Features

- 🎯 Placement prediction
- 📈 Placement probability
- 📊 Student performance analysis
- 💡 Personalized skill recommendations
- 📉 Skill performance visualization
- 🎨 Interactive Streamlit dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- GitHub

## 📂 Project Structure

```text
AI-Student-Placement-Prediction/
│
├── app.py
├── placement_prediction.py
├── generate_dataset.py
├── placement_data.csv
└── requirements.txt

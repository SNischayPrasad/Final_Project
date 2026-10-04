# Iris Flower Classification – AI Project Deployment

**Week 4 Capstone | AI Project Deployment**  
End-to-end Machine Learning project: train → serialize → deploy as a Flask web application.

**Author:** S. Nischay Prasad  
**GitHub:** [https://github.com/SNischayPrasad/Final_Project](https://github.com/SNischayPrasad/Final_Project)

---

## Project Overview

This project builds and deploys a classic **Iris flower classifier**.  

A Logistic Regression model is trained on the well-known Iris dataset, saved using Joblib, and served through a Flask web application. Users can enter sepal and petal measurements in a simple web UI and receive the predicted species (Setosa, Versicolor, or Virginica) along with confidence scores.

### Objectives Achieved
1. End-to-end ML pipeline (preprocessing → training → evaluation → serialization)
2. Model serialization with Joblib
3. REST-style prediction API + user-friendly web interface using Flask
4. Clean project structure and documentation

---

## Features

- **Machine Learning Pipeline** using `scikit-learn` (`StandardScaler` + `LogisticRegression`)
- **Model Persistence** with Joblib (model + feature names + target names)
- **Flask Web Application** with:
  - Clean HTML form for input
  - Real-time prediction via `/predict` endpoint
  - Confidence scores and class probabilities
  - Health-check endpoint (`/health`)
- Responsive and simple UI

---

## Project Structure

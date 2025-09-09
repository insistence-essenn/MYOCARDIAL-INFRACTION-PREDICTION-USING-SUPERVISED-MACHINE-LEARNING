# 🫀 Myocardial Infarction Prediction Using Supervised Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Django](https://img.shields.io/badge/Django-3.2+-green.svg)](https://www.djangoproject.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **A comprehensive machine learning solution for predicting myocardial infarction (heart attack) risk using clinical data and advanced supervised learning algorithms.**

## 🌟 Overview

This project provides an end-to-end machine learning solution for predicting myocardial infarction risk. It combines data preprocessing, model training, evaluation, and deployment through both Jupyter notebooks for research and a Django web application for real-world usage.

### ✨ Key Features

- 🤖 **Multiple ML Models**: Logistic Regression, Neural Networks, Random Forest
- 📊 **Comprehensive Analysis**: 6 detailed Jupyter notebooks covering the full ML pipeline
- 🌐 **Web Application**: User-friendly Django interface for predictions
- 📈 **Data Visualization**: Rich charts and graphs for data exploration
- 🔒 **User Authentication**: Secure login system for patients and doctors
- 📝 **Feedback System**: Doctor review and feedback functionality
- 📱 **Responsive Design**: Modern, mobile-friendly interface

## 📋 Table of Contents

- [🏗️ Project Structure](#️-project-structure)
- [📊 Dataset Information](#-dataset-information)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [🚀 Installation](#-installation)
- [💻 Usage](#-usage)
- [🌐 Web Application](#-web-application)
- [📓 Jupyter Notebooks](#-jupyter-notebooks)
- [🎯 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🏗️ Project Structure

```
├── 📂 Deploy/
│   └── 📂 latest/
│       ├── 📂 new/              # Django app
│       │   ├── 📁 templates/    # HTML templates
│       │   ├── 📁 static/       # CSS, JS, images
│       │   ├── 🐍 models.py     # Database models
│       │   ├── 🐍 views.py      # Application logic
│       │   └── 🐍 forms.py      # User forms
│       └── ⚙️ manage.py         # Django management
├── 📓 M1.ipynb                  # Data Preprocessing
├── 📓 M2.ipynb                  # Exploratory Data Analysis
├── 📓 M3.ipynb                  # Logistic Regression Model
├── 📓 M4.ipynb                  # Neural Network Model
├── 📓 M5.ipynb                  # Random Forest Model
├── 📓 M6.ipynb                  # Model Comparison & Evaluation
├── 📊 heart.csv                 # Dataset (368 samples, 23 features)
├── 🤖 RF.pkl                    # Saved Random Forest Model
├── 📝 des.txt                   # Feature descriptions
└── 📖 README.md                 # This file
```

## 📊 Dataset Information

### 📈 Dataset Overview
- **Size**: 368 patient records
- **Features**: 23 clinical and demographic variables
- **Target**: Binary classification (Heart Attack Risk: Yes/No)

### 🏥 Clinical Features

| Feature | Description | Type |
|---------|-------------|------|
| **Age** | Patient age | Numeric |
| **Gender** | Patient sex (Male/Female) | Categorical |
| **Locality** | Living area (Rural/Urban) | Categorical |
| **Marital_status** | Marital status | Categorical |
| **Sleep** | Sleep quality issues | Binary |
| **Depression** | Depression status | Binary |
| **Smoking** | Smoking habits | Binary |
| **Diabetes** | Diabetes status | Binary |
| **BP** | Blood pressure | Numeric |
| **Hypersensitivity** | Allergic reactions | Binary |
| **cp** | Chest pain type (1-4) | Categorical |
| **trestbps** | Resting blood pressure (mm Hg) | Numeric |
| **chol** | Serum cholesterol (mg/dl) | Numeric |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary |
| **restecg** | Resting ECG results (0-2) | Categorical |
| **thalach** | Maximum heart rate achieved | Numeric |
| **exang** | Exercise induced angina | Binary |
| **oldpeak** | ST depression | Numeric |
| **slope** | Slope of peak exercise ST segment | Numeric |
| **ca** | Number of major vessels | Numeric |
| **thal** | Thalassemia | Numeric |
| **Mortality** | Target variable | Binary |

## 🤖 Machine Learning Models

### 📈 Implemented Algorithms

1. **📊 Logistic Regression** (M3.ipynb)
   - Linear classification model
   - Interpretable coefficients
   - Fast training and prediction

2. **🧠 Neural Network** (M4.ipynb)
   - Multi-layer perceptron classifier
   - Non-linear pattern recognition
   - Advanced feature learning

3. **🌳 Random Forest** (M5.ipynb)
   - Ensemble learning method
   - Feature importance analysis
   - Robust to overfitting
   - **Currently deployed in web app**

### 🎯 Model Performance

Each model is evaluated using:
- ✅ **Accuracy Score**
- 📊 **Confusion Matrix**
- 📋 **Classification Report**
- 📈 **ROC Curves** (where applicable)

## 🚀 Installation

### 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/insistence-essenn/MYOCARDIAL-INFRACTION-PREDICTION-USING-SUPERVISED-MACHINE-LEARNING.git
   cd MYOCARDIAL-INFRACTION-PREDICTION-USING-SUPERVISED-MACHINE-LEARNING
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   pip install django jupyter plotly joblib
   ```

4. **Set up Django application**
   ```bash
   cd Deploy/latest
   python manage.py makemigrations
   python manage.py migrate
   python manage.py createsuperuser  # Optional: create admin user
   ```

## 💻 Usage

### 🌐 Web Application

1. **Start the Django server**
   ```bash
   cd Deploy/latest
   python manage.py runserver
   ```

2. **Access the application**
   - Open your browser and go to: `http://127.0.0.1:8000`
   - Login as user or doctor
   - Fill in patient information
   - Get instant heart attack risk prediction

### 📓 Jupyter Notebooks

1. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Run notebooks in order**:
   - `M1.ipynb` - Data preprocessing and cleaning
   - `M2.ipynb` - Exploratory data analysis and visualization
   - `M3.ipynb` - Logistic regression model training
   - `M4.ipynb` - Neural network implementation
   - `M5.ipynb` - Random forest model (production model)
   - `M6.ipynb` - Model comparison and final evaluation

## 🌐 Web Application

### 🔐 User Roles

- **👤 Patients**: Input medical data and receive predictions
- **👨‍⚕️ Doctors**: Review patient data and provide feedback

### 🎨 Features

- **🔒 Secure Authentication**: Login system for users and doctors
- **📝 Interactive Forms**: Easy-to-use input forms for medical data
- **⚡ Real-time Predictions**: Instant risk assessment using trained models
- **💬 Feedback System**: Doctor review and recommendation system
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

### 🖼️ User Interface

The web application features a clean, modern interface with:
- Intuitive navigation
- Clear form layouts
- Visual feedback for predictions
- Professional medical styling

## 📓 Jupyter Notebooks

### 📖 Detailed Breakdown

| Notebook | Purpose | Key Components |
|----------|---------|----------------|
| **M1.ipynb** | Data Preprocessing | Data cleaning, handling missing values, feature encoding |
| **M2.ipynb** | EDA & Visualization | Statistical analysis, correlation matrices, distribution plots |
| **M3.ipynb** | Logistic Regression | Linear model implementation, feature importance |
| **M4.ipynb** | Neural Network | MLP classifier, hyperparameter tuning |
| **M5.ipynb** | Random Forest | Ensemble model, feature selection, model optimization |
| **M6.ipynb** | Model Comparison | Performance metrics, final model selection |

## 🎯 Results

### 📊 Model Performance Summary

The project evaluates multiple machine learning algorithms to find the best performing model for myocardial infarction prediction. Key metrics include:

- **🎯 Accuracy**: Overall prediction correctness
- **🔍 Precision**: True positive rate
- **📈 Recall**: Sensitivity to positive cases
- **⚖️ F1-Score**: Balanced precision and recall

*Note: Specific performance metrics can be found in the individual notebook files.*

### 🏆 Production Model

The **Random Forest model** is currently deployed in the web application due to its:
- High accuracy and reliability
- Robustness to outliers
- Feature importance insights
- Balanced performance across all metrics

## 🔧 Technical Stack

- **🐍 Backend**: Python, Django
- **🤖 Machine Learning**: Scikit-learn, Pandas, NumPy
- **📊 Visualization**: Matplotlib, Seaborn, Plotly
- **🌐 Frontend**: HTML5, CSS3, Bootstrap
- **💾 Database**: SQLite (default Django)
- **📓 Development**: Jupyter Notebooks

## 🔮 Future Enhancements

- [ ] 🚀 Deploy to cloud platforms (AWS, Heroku, etc.)
- [ ] 📱 Mobile application development
- [ ] 🤖 Advanced deep learning models (CNN, LSTM)
- [ ] 📊 Real-time data integration
- [ ] 🔔 Alert systems for high-risk patients
- [ ] 📈 Advanced visualization dashboards
- [ ] 🌍 Multi-language support
- [ ] 🔗 API development for third-party integration

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

### 📝 Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add comments and documentation for new features
- Include tests for new functionality
- Update README if necessary

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Project Team** - Initial work and development

## 🙏 Acknowledgments

- Heart disease dataset contributors
- Scikit-learn community
- Django framework developers
- Medical professionals who provided domain expertise

## 📞 Support

If you have any questions or need support:

- 📧 Open an issue on GitHub
- 💬 Start a discussion in the repository
- 📖 Check the documentation in the notebooks

---

<div align="center">
  <b>⭐ If you found this project helpful, please give it a star! ⭐</b>
  <br><br>
  <i>Made with ❤️ for improving healthcare through technology</i>
</div>
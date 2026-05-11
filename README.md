# 🚦 Smart Road Traffic Accident Severity Prediction System

## 📌 Project Overview

This project focuses on analyzing and predicting road traffic accident severity using Machine Learning techniques and an interactive Streamlit dashboard.

The system performs:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Accident Severity Prediction
- Risk Classification
- Feature Importance Analysis
- Model Comparison
- SQL Database Integration
- Interactive Visualization Dashboard

The project helps:
- Traffic Management Authorities
- Smart City Planning Teams
- Hospitals & Emergency Services
- Insurance Companies
- Public Safety Departments

to identify accident-prone conditions and improve road safety strategies.

---

# ⚠️ Dataset Availability Notice

The original dataset used in this project belongs to the client and cannot be uploaded publicly due to privacy and confidentiality restrictions.

Therefore:
- Only the source code, ML pipeline, and Streamlit dashboard are shared in this repository.
- The dataset is intentionally excluded from GitHub.

Users can use any similar Road Traffic Accident dataset from Kaggle containing related features.

---

# 🔎 Suggested Kaggle Search Keywords

Search in Kaggle using:

```bash
Road Traffic Accident Dataset
```

or

```bash
Traffic Accident Severity Dataset
```

or

```bash
Road Accident Prediction Dataset
```

---

# 📂 Project Structure

```bash
RTA_PROJECT/
│
├── MAIN CODE FINAL.py
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── your_dataset.csv
│
├── output/
│   ├── rta_best_pipeline.pkl
│   ├── rta_feature_schema.json
│   ├── rta_model_metrics.csv
│   ├── rta_model_comparison.csv
│   ├── rta_model_comparison_f1.png
│   └── rta_model_comparison_heatmap.png
│
└── images/
    └── dashboard_preview.png
```

---

# 🎯 Objectives

- Analyze road traffic accident patterns
- Predict accident severity using ML algorithms
- Identify high-risk accident conditions
- Compare multiple machine learning models
- Visualize accident analytics interactively
- Store accident data in MySQL database
- Build an accident risk prediction dashboard

---

# 📊 Dataset Features

The project expects a dataset containing the following or similar features:

| Feature Name | Description |
|---|---|
| Time | Accident occurrence time |
| Day_of_week | Day when accident occurred |
| Age_band_of_driver | Driver age category |
| Sex_of_driver | Driver gender |
| Educational_level | Driver education level |
| Vehicle_driver_relation | Driver relation to vehicle |
| Driving_experience | Driving experience level |
| Type_of_vehicle | Vehicle category/type |
| Owner_of_vehicle | Vehicle ownership |
| Service_year_of_vehicle | Vehicle age/service years |
| Defect_of_vehicle | Vehicle defect condition |
| Area_accident_occured | Accident area/location |
| Lanes_or_Medians | Lane/median information |
| Road_allignment | Road alignment type |
| Types_of_Junction | Junction/intersection type |
| Road_surface_type | Road surface category |
| Road_surface_conditions | Road condition |
| Light_conditions | Lighting condition |
| Weather_conditions | Weather condition |
| Type_of_collision | Type of collision |
| Number_of_vehicles_involved | Vehicles involved in accident |
| Number_of_casualties | Casualties count |
| Vehicle_movement | Vehicle movement status |
| Casualty_class | Casualty classification |
| Sex_of_casualty | Casualty gender |
| Age_band_of_casualty | Casualty age group |
| Casualty_severity | Casualty severity |
| Work_of_casuality | Casualty occupation |
| Fitness_of_casuality | Fitness condition |
| Pedestrian_movement | Pedestrian movement |
| Cause_of_accident | Main cause of accident |
| Accident_severity | Target variable |

---

# 🎯 Target Variable

The machine learning model predicts:

```text
Accident_severity
```

Possible classes may include:

- Slight Injury
- Serious Injury
- Fatal Injury

---

# 🧠 Machine Learning Models Used

The project compares multiple classification models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

# ⚙️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- SQLAlchemy
- PyMySQL
- Joblib

---

# 🔍 Exploratory Data Analysis (EDA)

The project performs:

- Univariate Analysis
- Numerical Distribution Analysis
- Correlation Heatmaps
- Pairplots
- Outlier Detection
- Accident Pattern Visualization

---

# 📈 Business Decision Moments (BDMs)

The system identifies:

- Accident-prone areas
- Peak accident timings
- High-risk road conditions
- Weather-related accident risks
- Severity vs vehicle involvement relationships

These insights support:
- Traffic management
- Emergency response planning
- Smart city analytics
- Road safety improvement

---

# 🗄️ SQL Integration

The dataset is connected to a MySQL database using SQLAlchemy.

### Database Table

```sql
accident_data
```

---

# 🖥️ Streamlit Dashboard Features

The dashboard provides:

✅ CSV/XLSX Upload  
✅ Accident Severity Prediction  
✅ Risk Classification  
✅ Download Prediction Results  
✅ Feature Importance Visualization  
✅ Confusion Matrix  
✅ Classification Report  
✅ Histogram Visualization  
✅ Interactive Data Exploration  
✅ Prediction Filtering  

---

# 🚀 How to Run the Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/RTA-Accident-Prediction.git
cd RTA-Accident-Prediction
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Add Dataset

Download a similar dataset from Kaggle and place it inside the `dataset/` folder.

Update dataset path inside the code:

```python
CSV_PATH = Path("dataset/your_dataset.csv")
```

---

## 4️⃣ Run Main ML Pipeline

```bash
python "MAIN CODE FINAL.py"
```

---

## 5️⃣ Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 📊 Output Files Generated

The system generates:

- Trained ML Pipeline (.pkl)
- Feature Schema (.json)
- Model Metrics (.csv)
- Model Comparison Reports
- Heatmaps
- F1 Score Charts

---

# 🧪 Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# 📌 Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
EDA & Visualization
   ↓
Preprocessing
   ↓
SMOTE Balancing
   ↓
Model Training
   ↓
Evaluation & Comparison
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
Streamlit Deployment
```

---

# 📷 Dashboard Preview

Add your dashboard screenshots here:

```bash
images/dashboard_preview.png
```

---

# 👨‍💻 Author

## K SAI PRAGNA

### Project:
Smart Road Traffic Accident Severity Prediction System

---

# 📜 License

This project is developed for educational and research purposes only.

---

# ⭐ Future Enhancements

- Real-time Traffic API Integration
- GPS-based Accident Mapping
- IoT Sensor Integration
- Deep Learning Models
- Cloud Deployment
- Real-time Risk Monitoring Dashboard

---

# 🙌 Acknowledgements

- Scikit-learn
- Streamlit
- Open Source Community
# 🚦 Smart Accident Risk Classification Using Traffic and Road Condition Data

## 📌 Project Overview

This project focuses on predicting road traffic accident occurrence using Machine Learning techniques, Exploratory Data Analysis (EDA), SQL integration, and an interactive Streamlit dashboard.

The system analyzes multiple traffic-related, environmental, driver-related, and road-condition factors to classify whether an accident is likely to occur.

The project includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning Model Training
- Accident Risk Prediction
- Feature Importance Analysis
- SQL Database Integration
- Interactive Streamlit Dashboard
- Model Comparison & Evaluation

---

# ⚠️ Dataset Availability Notice

The original dataset used in this project belongs to the client and cannot be uploaded publicly due to privacy and confidentiality restrictions.

Therefore:

- Only the source code, ML pipeline, and Streamlit dashboard are uploaded in this repository.
- The original dataset is intentionally excluded from GitHub.

Users can use any similar Road Traffic Accident dataset from Kaggle containing related traffic and road-condition features.

---

# 🔎 Suggested Kaggle Search Keywords

Search on Kaggle using:

```bash
Road Traffic Accident Dataset
```

or

```bash
Traffic Accident Prediction Dataset
```

or

```bash
Road Accident Classification Dataset
```

---

# 📂 Project Structure

```bash
SMART_ACCIDENT_RISK_CLASSIFICATION/
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

# 🎯 Project Objectives

- Analyze traffic accident patterns
- Predict accident occurrence using ML algorithms
- Identify high-risk traffic conditions
- Compare multiple machine learning models
- Build an interactive accident risk dashboard
- Store accident data into MySQL database
- Improve road safety analytics

---

# 📊 Dataset Features

The project expects a dataset containing traffic accident-related attributes similar to the following:

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
| Service_year_of_vehicle | Vehicle service age |
| Defect_of_vehicle | Vehicle defect condition |
| Area_accident_occured | Accident area/location |
| Lanes_or_Medians | Lane/median details |
| Road_allignment | Road alignment type |
| Types_of_Junction | Junction/intersection type |
| Road_surface_type | Road surface category |
| Road_surface_conditions | Road surface condition |
| Light_conditions | Lighting condition |
| Weather_conditions | Weather condition |
| Type_of_collision | Collision type |
| Number_of_vehicles_involved | Number of vehicles involved |
| Number_of_casualties | Number of casualties |
| Vehicle_movement | Vehicle movement status |
| Casualty_class | Casualty classification |
| Sex_of_casualty | Casualty gender |
| Age_band_of_casualty | Casualty age group |
| Casualty_severity | Casualty severity |
| Work_of_casuality | Casualty occupation |
| Fitness_of_casuality | Fitness condition |
| Pedestrian_movement | Pedestrian movement |
| Cause_of_accident | Main cause of accident |

---

# 🎯 Target Variable

The machine learning model predicts:

```text
accident_occurred
```

The target variable represents whether an accident occurred or not under specific traffic, road, environmental, and driver-related conditions.

---

# 📌 Possible Classes

| Value | Meaning |
|---|---|
| 0 | No Accident |
| 1 | Accident Occurred |

---

# ⚠️ Important Dataset Note

The original client dataset contained a custom engineered target variable named:

```text
accident_occurred
```

Since the dataset is not publicly shared, users downloading alternative datasets from Kaggle may need to manually create the target column.

Example:

```python
df['accident_occurred'] = np.where(df['Accident_severity'] == 'Slight Injury', 0, 1)
```

You may customize the target logic based on your project requirements.

---

# 🧠 Machine Learning Models Used

The project compares multiple classification algorithms:

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
- Accident Risk Visualization

---

# 📈 Business Decision Moments (BDMs)

The system identifies:

- Accident-prone locations
- Peak accident timings
- Weather-related accident risks
- Road-condition risk analysis
- Vehicle involvement patterns

These insights help:
- Traffic authorities
- Smart city planners
- Emergency response teams
- Insurance companies

---

# 🗄️ SQL Integration

The dataset is connected to MySQL using SQLAlchemy.

### Database Table

```sql
accident_data
```

---

# 🖥️ Streamlit Dashboard Features

The interactive dashboard provides:

✅ CSV/XLSX Upload  
✅ Accident Prediction  
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
git clone https://github.com/your-username/Smart-Accident-Risk-Classification.git
cd Smart-Accident-Risk-Classification
```

---

## 2️⃣ Install Requirements

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
- Comparison Reports
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

Add your dashboard screenshots inside:

```bash
images/dashboard_preview.png
```

---

# 👨‍💻 Author

## K SAI PRAGNA

### Project:
Smart Accident Risk Classification Using Traffic and Road Condition Data

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
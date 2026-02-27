# Absenteeism Analysis Dashboard

## Project Overview

This project leverages data mining techniques to analyze employee absenteeism patterns in a Brazilian courier company. The goal was to segment employees into distinct risk groups using unsupervised learning, and to develop targeted HR strategies based on shared characteristics and absence patterns. The dashboard serves as an HR tool to determine the absenteeism risk level of potential new hires and provides actionable insights for HR professionals.

## Project Steps

### 1. Data Loading and Exploration

- The dataset was loaded and explored to understand each feature, detect data quality issues (missing values, skewness, outliers), and identify patterns for clustering.
- **Exploratory Data Analysis (EDA)** was performed to visualize key statistics and distributions, gaining insights into absenteeism drivers and identifying potential features for clustering.

### 2. Data Pre-processing

- **Missing Value Handling**: Missing values in categorical features were handled by forward filling or filling with a constant, while temporal features were managed by dropping rows with excessive missingness.
- **Outlier Handling**: Extreme values in features like Height and Transportation Expense were capped to preserve data integrity.
- **Fixing Inconsistent Data**: Categorical labels were standardized, and data types were converted for consistency.
- **Feature Engineering**: New features were created, including:
  - **Age** from Date of Birth.
  - **Commute Burden Index**: Combines transportation expense, distance, and commute time.
  - **Home Responsibility Index**: Sum of number of children and pets.
- **Data Transformation**: Features were scaled and encoded for model compatibility, ensuring that the data was suitable for clustering and classification algorithms.

### 3. Clustering and Model Training

- **Unsupervised Clustering**: K-Means, Hierarchical Clustering, and Gaussian Mixture Models were compared to segment employees into clusters. The best-performing model was selected based on silhouette scores and cluster interpretability.
- **Cluster Profiling**: Each cluster was profiled to understand the characteristics of high-risk and low-risk employees, providing actionable insights for HR.
- **Model Training**: A Random Forest Classifier was trained to predict cluster membership for new employees based on their features, leveraging supervised learning for accurate predictions.

### 4. Key Features

- **Cluster Profiles**: Employees are classified into three risk groups:
  - **Long-Distance Commuters (Low Risk)**
  - **Experienced Urban Workers (High Risk)**
  - **Young Family-Oriented (Moderate Risk)**
- **HR Insights**: For each cluster, the dashboard provides tailored HR insights and guidelines to help HR teams address the unique challenges and needs of each group.

### 5. HR New Employee Clustering Tool

- The "New Employee Clustering Tool" was developed for HR use. It is deployed at [https://absentee-analysis.vercel.app/](https://absentee-analysis.vercel.app/).
- This enables HR teams to proactively address the unique challenges and needs of each group, supporting targeted onboarding, retention strategies, and workplace interventions.

## Data Mining Aspects

- **Exploratory Data Analysis (EDA)**: Comprehensive EDA was performed to understand the dataset and identify patterns for clustering.
- **Feature Engineering**: New features were created to capture key aspects of employee absenteeism, such as the Commute Burden Index and Home Responsibility Index.
- **Unsupervised Clustering**: Multiple clustering algorithms were compared to segment employees into distinct risk groups.
- **Model Evaluation**: The best-performing clustering model was selected based on silhouette scores and cluster interpretability.
- **Supervised Learning**: A Random Forest Classifier was trained to predict cluster membership for new employees, leveraging supervised learning for accurate predictions.

This project showcases the ability to build and deploy a practical, data-driven solution for business intelligence and HR applications, highlighting the use of advanced data mining techniques throughout the process.

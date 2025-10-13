

# Employee Absenteeism Analysis and Clustering  
**Group 10 - DMI 2526**
**Members:** Tyler Johnston, Fariha Khan, Zara Nayami Lelis de Carvalho, and Victory Amakekemi

---

## Project Overview

This project aims to analyze employee absenteeism, identify meaningful worker segments through clustering, and provide actionable insights to support Human Resources decision-making. The project consists of three main phases:  

1. **Exploratory Data Analysis (EDA) & Preprocessing**  
2. **Worker Segmentation through Clustering**  
3. **Knowledge in Action: Translating Insights into Practical Solutions**

---

## Dataset

- **Source:** The original unedited dataset is <i>absenteeism_data.csv</i>
- **Description:** The dataset contains employee information relevant to absenteeism; such as, the reason for absense, the month of absense, their date of birth, their number of children, being a social smoker, etc.
- **Size:** 801 rows with 22 columns 
---

## 1. Exploratory Data Analysis (EDA) & Preprocessing

**Objective:** Understand the dataset, address data quality issues, and prepare features for clustering.  

**Steps performed:**  
- **Data Cleaning:**  
  - Handled missing values using [method: mean, median, interpolation, etc.]  
  - Detected and treated outliers with [method: IQR, Z-score, etc.]  
- **Feature Engineering:**  
  - Created new variables to capture relevant absenteeism behaviors ([e.g., absenteeism_rate, overtime_hours_ratio])  
- **Data Transformation:**  
  - Encoded categorical features using [method: one-hot, label encoding]  
  - Normalized/scaled numerical features using [method: MinMaxScaler, StandardScaler]  
- **Insights:**  
  - Distribution analysis of key features  
  - Correlation between absenteeism and other features  
  - Identification of potential patterns for clustering

---

## 2. Worker Segmentation

**Objective:** Apply clustering techniques to segment employees into meaningful groups.  

**Clustering Process:**  
- **Feature Selection:**  
  - Included features: [list final features used]  
  - Excluded features: [list irrelevant or redundant features]  
- **Algorithms Explored:**  
  - K-Means  
  - Hierarchical Clustering  
  - DBSCAN / Other methods considered  
- **Model Selection:**  
  - Evaluated clustering quality using [metrics: silhouette score, Davies-Bouldin index, etc.]  
  - Chose final clustering model based on [criteria]  
- **Cluster Interpretation:**  
  - Described characteristics of each cluster  
  - Identified differences in absenteeism patterns  
  - Suggested potential HR strategies for each group

---

## 3. Knowledge in Action

**Objective:** Translate the analysis into actionable insights for the company.  

**Implemented Solutions:**  
- **Visual Dashboard:**  
  - TODO
- **Predictive Assignment:**  
  - TODO
- **Feature Importance Analysis:**  
  - TODO 
- **Recommendations:**  
  - TODO

---

## Project Files

| File | Description |
|------|-------------|
| `Group10_DMI_2526.ipynb` | Jupyter Notebook containing all analysis, preprocessing, and clustering code. Commented code includes exploration and decision-making steps. |
| `Group10_DMI_2526_report.pdf` | Structured report summarizing analytical processes, results, and practical recommendations (max 15 pages). |

---

## Evaluation Criteria Addressed

- **Notebook Clarity:** Clear goals, comments, and insights linked to next steps  
- **Data Exploration:** Relevant visualizations, distributions, and correlations  
- **Data Preprocessing:** Complete handling of missing data, outliers, encoding, and scaling  
- **Clustering:** Well-justified feature selection, modeling, evaluation, and interpretation  
- **Knowledge in Action:** Creative, actionable, and well-explained solutions for HR  
- **Report Quality:** Coherent storytelling linking methodology to results and conclusions

---

## How to Run

1. Clone this repository:  
   ```bash
   git clone <repository-url>
2. Open the notebook:
    ```bash
    jupyter notebook Group10_DMI_2526.ipynb
3. Follow the notebook cells sequentially to reproduce the analysis and visualizations.
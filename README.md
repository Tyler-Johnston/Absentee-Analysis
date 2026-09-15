# Absenteeism Analysis Dashboard

> An end-to-end data-mining project that turns employee absenteeism data into actionable workforce profiles and an interactive decision-support prototype.

[Live demo](https://absentee-analysis.vercel.app/) · [Explore the analysis notebook](Group10_DMI_2526.ipynb)

## The project

A Brazilian courier company wants to better understand its employees' absenteeism patterns. Rather than treating every absence as the same problem, this project uses unsupervised learning to identify groups of employees with similar demographic, work, commute, and lifestyle characteristics.

The result is a three-segment employee profile framework, paired with a Flask web application that assigns a hypothetical employee profile to a segment and presents relevant workplace-support recommendations.

## Results at a glance

| Employee profile | Share of dataset | Average absence | What stands out |
| --- | ---: | ---: | --- |
| Long-Distance Commuters | 27.1% | 5.40 hours | Lowest absenteeism despite the highest commute burden |
| Experienced Urban Workers | 37.4% | 8.46 hours | Highest absenteeism despite the shortest commute |
| Young Family-Oriented | 35.5% | 7.04 hours | Near-average absenteeism with the highest home responsibilities |

The central finding challenges a simple commute-based explanation: the group with the greatest commute burden had the lowest average absenteeism, while the short-commute group had the highest. This points to workplace, health, engagement, or other contextual factors worth investigating rather than assuming commuting is the primary driver.

## From data to decision support

```text
Raw employee data
        ↓
Cleaning and exploratory analysis
        ↓
Feature engineering and scaling
        ↓
Cluster evaluation and employee segmentation
        ↓
Random Forest profile-assignment model
        ↓
Flask employee-profile support tool
```

## Methodology

### Dataset and exploration

The analysis uses **800 employee records** with **22 source variables**, covering absenteeism, demographics, work conditions, commute characteristics, and lifestyle indicators. Exploratory data analysis examined distributions, missing values, outliers, inconsistent labels, and relationships between absenteeism and potential drivers.

### Preparation and feature engineering

The pipeline addressed missing values, capped selected extreme values, standardized inconsistent categorical labels, and scaled continuous features before clustering.

Two composite features were created to make the analysis more interpretable:

- **Commute Burden Index** combines transportation expense, distance from home to work, and estimated commute time.
- **Home Responsibility Index** combines the number of children and pets.

The final segmentation used **15 features** that describe employee characteristics. Calendar variables, absence reasons, and absenteeism hours were intentionally excluded from the clustering inputs so the groups represent *who employees are*, rather than when or why an absence occurred.

### Clustering and validation

K-Means, Ward hierarchical clustering, average-linkage hierarchical clustering, and Gaussian mixture models were compared using three clusters. K-Means was selected because it tied for the strongest silhouette score while producing the clearest, most actionable profiles for HR.

| Model | Silhouette score |
| --- | ---: |
| **K-Means** | **0.561** |
| Hierarchical (Ward) | 0.557 |
| Hierarchical (Average) | 0.518 |
| Gaussian Mixture | 0.561 |

Although two clusters achieved the highest raw silhouette score (0.668), three clusters provided a more useful balance of separation, interpretability, and distinct intervention strategies.

### Deployment model

After clustering, a Random Forest classifier was trained to reproduce cluster assignments for new profile scenarios. Five-fold cross-validation achieved **99.86% accuracy (±0.28%)**. This score measures how well the classifier recreates the discovered cluster labels—it is not a prediction of whether an individual employee will be absent.

## Interactive prototype

The Flask application packages the analysis into a simple form-based prototype. A user supplies a hypothetical employee profile; the tool then:

1. Calculates the two engineered indices.
2. Applies the same scaling used in the analysis pipeline.
3. Assigns the profile to one of the three segments.
4. Displays the segment's typical absenteeism level and tailored workplace-support ideas.

Suggested interventions include flexible schedules, occasional remote work where suitable, wellbeing support, recognition programmes, and family-friendly policies.

## Responsible use

This is an academic data-mining and decision-support prototype. Its segments describe patterns in an anonymized historical dataset; they do **not** establish why a specific person is absent, predict an individual's future attendance, or determine employment eligibility.

The prototype must not be used to screen candidates, make hiring decisions, or automate employment decisions. Several input attributes—such as age, body mass index, family status, and lifestyle indicators—can be sensitive or legally protected in employment contexts. Any real-world application would require legal review, fairness testing, data-governance controls, and meaningful human oversight.

## Tech stack

- **Analysis:** Python, pandas, NumPy, Matplotlib, Seaborn
- **Machine learning:** scikit-learn (K-Means, hierarchical clustering, Gaussian mixture models, Random Forest)
- **Application:** Flask, HTML, CSS
- **Deployment:** Vercel

## Repository structure

```text
.
├── Group10_DMI_2526.ipynb       # EDA, preprocessing, clustering, and evaluation
├── absenteeism_data.csv         # Source dataset
├── requirements.txt             # Analysis environment dependencies
└── hr_analysis_tool/
    ├── app.py                   # Flask application
    ├── templates/               # Form and result views
    ├── static/                  # Application styling
    ├── *.pkl                    # Saved classifier, scaler, columns, and parameters
    └── requirements.txt         # Application dependencies
```

## Run locally

### Analysis notebook

```bash
pip install -r requirements.txt
jupyter notebook Group10_DMI_2526.ipynb
```

### Flask prototype

```bash
cd hr_analysis_tool
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Team

Developed for the Data Mining I course by Fariha Khan, Tyler Johnston, Zara Carvalho, and Victory Amakekemi.

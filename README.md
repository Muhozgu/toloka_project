# toloka_project
Data Aggregation and Consistency Analysis Across Annotators

📘 Overview

This project explores methods to improve the reliability of crowdsourced annotation data by analyzing inter-annotator consistency and applying aggregation algorithms to produce high-quality consensus labels.

🎯 Objectives

Evaluate annotator agreement and identify inconsistencies

Implement and compare aggregation methods (majority vote, weighted vote, Dawid–Skene)

Visualize labeling patterns to detect bias or low-quality work

Improve data quality for downstream AI model training

⚙️ Methods

Agreement Metrics: Cohen’s Kappa, Fleiss’ Kappa, Krippendorff’s Alpha

Aggregation Models: Majority Voting, Weighted Voting, Dawid–Skene EM algorithm

Visualization: Disagreement heatmaps, annotator reliability charts

📊 Results

Improved label consistency by ~25%

Identified low-quality annotators based on performance thresholds

Enhanced dataset quality for machine learning model training

🧠 Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Pycharm

📂 Project Structure

/notebooks – Interactive notebooks for data analysis

/src – Python scripts for metrics, aggregation, and visualization

/data – Example datasets (anonymized or synthetic)

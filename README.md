# MEDIGUIDE

# Introduction
Cerebral stroke is a critical medical condition characterized by an abrupt disruption in blood flow to the brain. The lack of blood flow deprives brain cells of oxygen and essential nutrients, which can result in cell death, brain damage, disability, and even death. Prompt detection and intervention are crucial to minimizing brain damage and improving patient outcomes.

This project aims to address the challenge of predicting brain stroke using machine learning models, specifically focusing on highly imbalanced datasets that are common in medical diagnoses.

# Problem Statement
In this project, we tackle the issue of accurately predicting brain strokes using a dataset that contains 43,400 samples with only 783 stroke cases (a mere 1.804% of the total data). This extreme class imbalance poses significant challenges for traditional machine learning algorithms, which tend to favor the majority class, leading to poor performance in detecting the minority class (i.e., stroke cases).

Our objective is to build a model that can handle this imbalance effectively and predict stroke cases with high accuracy, addressing the specific needs of medical diagnosis where false negatives can have serious consequences.

# Dataset
The dataset used for this project consists of 12 features, including various patient health metrics such as age, gender, hypertension, heart disease, glucose levels, and more. Out of 43,400 patient records, only 783 patients experienced a stroke, making stroke detection particularly difficult.

Key Dataset Characteristics:
Total samples: 43,400
Stroke samples: 783 (1.804% of total)
Features: 12 patient health attributes

# Challenges
The primary challenge of this project is the significant class imbalance, where stroke cases constitute less than 2% of the dataset. Machine learning models trained on such datasets often struggle to detect minority classes effectively. Without proper handling, this imbalance results in poor model performance, particularly in terms of precision, recall, and AUC-ROC.

# Key challenges include:

Class Imbalance: Standard algorithms struggle to detect rare events like strokes.
Avoiding Overfitting: Balancing the data without introducing bias or overfitting to synthetic samples.
Generalization: Ensuring the model generalizes well to unseen data and reduces false negatives.
Methodology
To address the class imbalance problem and improve the predictive performance of stroke detection, we used a multi-step approach that involved data preprocessing, balancing techniques, and model building.

Data Preprocessing
Missing Data Handling: Imputation techniques were used for any missing data points in the dataset.
Feature Scaling: Standard scaling was applied to numeric features to normalize the data.
Feature Selection: Only the most relevant features were retained based on exploratory data analysis.
Balancing the Dataset
We employed various techniques to address the class imbalance, focusing on both oversampling and undersampling strategies:

Oversampling: Increasing the instances of the minority class (stroke cases) using techniques like:

SMOTE (Synthetic Minority Oversampling Technique)
Borderline SMOTE
SVM-SMOTE
SMOTE-TOMEK (combining SMOTE and Tomek Links for better separation)
SMOTE-ENN (combining SMOTE and Edited Nearest Neighbors)
SMOTEN (a novel approach proposed by us that enhances the effectiveness of SMOTE)
Undersampling: Reducing the number of samples from the majority class.

Model Building
After preprocessing and balancing the dataset, we applied various machine learning algorithms to build predictive models. These models include:

Artificial Neural Networks (ANN)
Random Forest Classifier
K-Nearest Neighbors (KNN)
Logistic Regression
Support Vector Machine (SVM)
Naive Bayes
AdaBoost
To further enhance performance, we used an Ensemble Learning Voting Classifier, combining the best-performing models:

Random Forest: 99% accuracy
KNN: 98% accuracy
By combining these models, we achieved the highest possible accuracy for stroke detection.

# Results
After applying various balancing techniques and machine learning algorithms, we obtained the following results:

Best Accuracy: 99% using the Voting Classifier (Random Forest + KNN)
AUC-ROC: 0.99, indicating a strong ability to distinguish between stroke and non-stroke cases.
Baseline Model (ANN): Achieved 92% accuracy.
Performance Comparison:
Model	Accuracy (%)	AUC-ROC
Random Forest	99%	0.99
K-Nearest Neighbors	98%	0.98
Artificial Neural Network (ANN)	92%	0.91
Logistic Regression	81%	0.79
AdaBoost	89%	0.88
Naive Bayes	60%	0.58
Support Vector Machine (SVM)	91%	0.90

# Conclusion
Through this project, we successfully addressed the challenge of brain stroke prediction in a highly imbalanced dataset. By applying a range of balancing techniques and leveraging the power of ensemble learning, we achieved a highly accurate and reliable model that can be used for early detection of brain strokes.

This approach demonstrates how machine learning techniques can be used to solve real-world problems in the medical field, especially when faced with imbalanced data. Early and accurate detection of strokes is vital to improving patient outcomes, and this project provides a robust solution for this critical task.

# VMWare-Case-Study
PCA and Logistic Regression

The main goal of this project is to predict whether customers of VMWare will take any targeted digital
actions, including Hands-on Lab (HoL), Evaluation, Webinar Registration, Seminar Registration, and Downloads.
The full dataset contains over 50,000 instances and it was split into training fold dataset, testing fold dataset as well as validation dataset.

Steps I took:
1. Handled missing values: for numeric features, imputation was performed, replacing missing values with each feature’s respective mean value. 
Missing values in categorical features were replaced with “None”, to avoid running into any issues with NA values.

2. Removed highly correlated numerical features and created dummy variables for categorical features.

3. Preformed PCA and kept 33 principal components as they will keep 95% of the variance of the original dataset

4. Trained the simple logistic regression model and assessed the performance using AUC.


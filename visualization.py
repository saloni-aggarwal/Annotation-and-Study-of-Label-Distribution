"""
This code performs the visualization by finding the covariance matrix and cohen's kappa value for each column between
two annotated sheets
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt

# Reading the two excel sheet with sheet title as 'Data to Annotate'
excel1 = pd.ExcelFile('JobQ3a_sna2485.xlsx')
excel2 = pd.ExcelFile('JobQ3a_lm7819.xlsx')
data1 = pd.read_excel(excel1, 'Data to Annotate')
data2 = pd.read_excel(excel2, 'Data to Annotate')
# Converting the data into csv files
data1.to_csv('dataset1.csv', index=False)
data2.to_csv('dataset2.csv', index=False)
# Reading the two csv files and skipping the initial spaces if any
dataset1 = pd.read_csv('dataset1.csv', encoding='unicode_escape', skipinitialspace=True)
dataset2 = pd.read_csv('dataset2.csv', encoding='unicode_escape', skipinitialspace=True)

# Getting only the columns for 3rd to 14th column in both the dataframes
df1 = dataset1[data1.columns[2:14]]
df2 = dataset2[data2.columns[2:14]]
# Getting column names
columns = df1.columns
# For every column do the following
for col in columns:
    # Converting the NaN values as 0 and any string values to numeric value
    df1[col] = pd.to_numeric(df1[col]).fillna(0)
    df2[col] = pd.to_numeric(df2[col]).fillna(0)
    # Making the confusion matrix for the column
    cm = confusion_matrix(df1[col], df2[col])
    # Getting the unique values that is 0 and 1
    labels = np.unique(df1[col])
    # Plotting and displaying the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()
    # Calculating and displaying the cohen's kappa score
    cohen = cohen_kappa_score(df1[col], df2[col])
    print(cohen)

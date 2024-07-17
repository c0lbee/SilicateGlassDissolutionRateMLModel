import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt



# Main data reading
data = pd.read_csv('dataset.csv')

#Drop uneeded columns
data = data.drop(columns = ['sum_poly', 'sum_poly.1', 'Measurement', '#','System Size (# atoms)', 'a (Å)', 'Vol (A3)', 'SUM'])

#Replaces null values with 0
data.replace([np.nan], 0, inplace=True)

# Removes rows with vandium due to small data with vandium
data = data[data['V2O5_mol'] == 0]



#This sections turns information into O bond percentages
Obonds = pd.DataFrame()

# TURNING BONDS INTO O-X FORM
Obonds['Al-O'] = data['Al-B']/2 + data['Al-Si']/2 + data['Al-Zr']/2 + data['Al-Al']
Obonds['B-O'] = data['Al-B']/2 + data['B-Si']/2 + data['B-Zr']/2 + data['B-B']
Obonds['Si-O'] = data['Al-Si']/2 + data['B-Si']/2 + data['Si-Zr']/2 + data['Si-Si']
Obonds['Zr-O'] = data['Al-Zr']/2 + data['B-Zr']/2 + data['Si-Zr']/2 + data['Zr-Zr']

#CALUCLATING FNET WITH MOLECULAR BOND ENTHALPY
Obonds['Fnet'] = Obonds['Al-O']*101 + Obonds['B-O']*119 + Obonds['Si-O']*106 + Obonds['Zr-O']*81
Obonds['pH (90 °C)'] = data['pH (90 °C)']
Obonds['ln r0'] = data['ln r0']



#Final data selection to train models
model = LinearRegression()
phLevels = data['pH (90 °C)'].unique()

for i in phLevels:
    if (i == 10 or i == 2): continue
    ln_r0 = data[data['pH (90 °C)'] == i]['ln r0']
    fnets = Obonds[Obonds['pH (90 °C)'] == i]['Fnet']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(fnets.values.reshape(-1, 1), ln_r0, test_size=0.2, random_state=42)
    
    # Train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    
    
    #Extends line to fit graph
    x_trainC = np.append(X_train, [fnets.max(),fnets.min()])
    
    # Get predictions for the min and max values
    min_pred = model.predict([[fnets.min()]])
    max_pred = model.predict([[fnets.max()]])
    
    # Ensure y_pred_train is a flattened array if it is not
    y_pred_train = y_pred_train.flatten()
    
    # Append the new predictions to y_pred_train
    y_predC = np.append(y_pred_train, [max_pred,min_pred])
    
    
    
    print(f'Mean Squared Error (Train): {mse_train}')
    print(f'Mean Squared Error (Test): {mse_test}\n')
    
    
    
    # Scatter plot with the regression line for the training data
    plt.scatter(X_train, y_train, color='blue', label=f'Training Data\nMSE: {mse_train:.2f}', marker='s', s=50)
    plt.scatter(X_test, y_test, color='green', label=f'Testing Data\nMSE: {mse_test:.2f}', s=50)
    plt.plot(x_trainC, y_predC, color='red', label='Regression Line')
    
    # Add titles and labels
    plt.title(f'Scatter Plot with Regression Line PH = {round(i,3)}')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.legend()
    
    # Show the plot
    plt.show()

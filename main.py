#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Data: Age, Alcohol Use, Marijuana Use
ages = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22.5, 24.5, 27.5, 32, 42, 57, 72])
alcohol_use = np.array([3.9, 8.5, 18.1, 29.2, 40.1, 49.3, 58.7, 64.6, 69.7,
                        83.2, 84.2, 83.1, 80.7, 77.5, 75, 67.2, 49.3])
marijuana_use = np.array([1.1, 3.4, 8.7, 14.5, 22.5, 28, 33.7, 33.4, 34,
                          33, 28.4, 24.9, 20.8, 16.4, 10.4, 7.3, 1.2])

# Split data into three categories
ages_12_20 = ages[ages <= 20]
ages_21_plus = ages[ages >= 20]

alcohol_12_20 = alcohol_use[ages <= 20]
alcohol_21_plus = alcohol_use[ages > 20]

marijuana_12_20 = marijuana_use[ages <= 20]
marijuana_21_plus = marijuana_use[ages >= 20]

# Function to perform linear regression using least squares
#Ax = b
#when it is not possible to solve for the coeffecients, least square approximation finds best fit line
#The Equation is (A^T * A)^-1 * A^T * Y
def least_squares(x, y): # 
    A = np.vstack((np.ones_like(x), x)).T  # Design matrix. First row is all 1's and second is X Values. We first make it a row vector and then transpose it to be a column vector
    firstTerm = A.T @ A 
    invertedFirstTerm = np.linalg.inv(firstTerm)
    coefficients = invertedFirstTerm @ A.T @ y  # Solve for coefficients
    return coefficients

# Function to plot marijuana data with regression lines
def plotMarijuanaData():
    plt.figure(figsize=(15, 8))
    
    # Full data
    coefficients_full = least_squares(ages, marijuana_use)
    intercept_full, slope_full = coefficients_full
    y_full_pred = intercept_full + slope_full * ages

    plt.subplot(1, 3, 1)
    plt.scatter(ages, marijuana_use, color='blue', label='Full Data')
    plt.plot(ages, y_full_pred, color='red', label=f'Fit: y = {slope_full:.2f}x + {intercept_full:.2f}')
    plt.title('Marijuana Use (All Ages)', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Marijuana Use', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Ages 12-20
    coefficients_12_20 = least_squares(ages_12_20, marijuana_12_20)
    intercept_12_20, slope_12_20 = coefficients_12_20
    y_12_20_pred = intercept_12_20 + slope_12_20 * ages_12_20

    plt.subplot(1, 3, 2)
    plt.scatter(ages_12_20, marijuana_12_20, color='green', label='Ages 12-20')
    plt.plot(ages_12_20, y_12_20_pred, color='red', label=f'Fit: y = {slope_12_20:.2f}x + {intercept_12_20:.2f}')
    plt.title('Marijuana Use (Ages 12-20)', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Marijuana Use', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Ages 21+
    coefficients_21_plus = least_squares(ages_21_plus, marijuana_21_plus)
    intercept_21_plus, slope_21_plus = coefficients_21_plus
    y_21_plus_pred = intercept_21_plus + slope_21_plus * ages_21_plus

    plt.subplot(1, 3, 3)
    plt.scatter(ages_21_plus, marijuana_21_plus, color='red', label='Ages 21+')
    plt.plot(ages_21_plus, y_21_plus_pred, color='blue', label=f'Fit: y = {slope_21_plus:.2f}x + {intercept_21_plus:.2f}')
    plt.title('Marijuana Use (Ages 21+)', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Marijuana Use', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to plot alcohol data with regression lines
def plotAlcoholData():
    plt.figure(figsize=(15, 8))
    
    # Full data
    coefficients_full = least_squares(ages, alcohol_use)
    intercept_full, slope_full = coefficients_full
    y_full_pred = intercept_full + slope_full * ages

    plt.subplot(1, 3, 1)
    plt.scatter(ages, alcohol_use, color='blue', label='Full Data')
    plt.plot(ages, y_full_pred, color='red', label=f'Fit: y = {slope_full:.2f}x + {intercept_full:.2f}')
    plt.title('Alcohol Use (All Ages)', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Alcohol Use', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Ages 12-20
    coefficients_12_20 = least_squares(ages_12_20, alcohol_12_20)
    intercept_12_20, slope_12_20 = coefficients_12_20
    y_12_20_pred = intercept_12_20 + slope_12_20 * ages_12_20

    plt.subplot(1, 3, 2)
    plt.scatter(ages_12_20, alcohol_12_20, color='green', label='Ages 12-20')
    plt.plot(ages_12_20, y_12_20_pred, color='red', label=f'Fit: y = {slope_12_20:.2f}x + {intercept_12_20:.2f}')
    plt.title('Alcohol Use (Ages 12-20)', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Alcohol Use', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Ages 21+
    coefficients_21_plus = least_squares(ages_21_plus, alcohol_21_plus)
    intercept_21_plus, slope_21_plus = coefficients_21_plus
    y_21_plus_pred = intercept_21_plus + slope_21_plus * ages_21_plus

    plt.subplot(1, 3, 3)
    plt.scatter(ages_21_plus, alcohol_21_plus, color='red', label='Ages 21+')
    plt.plot(ages_21_plus, y_21_plus_pred, color='blue', label=f'Fit: y = {slope_21_plus:.2f}x + {intercept_21_plus:.2f}')
    plt.title('Alcohol Use (Ages 21+)', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Alcohol Use', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Call function to plot the data
plotMarijuanaData()
#plotAlcoholData()

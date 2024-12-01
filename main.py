import numpy as np
import matplotlib.pyplot as plt

# Define the three data sets
data_sets = [
    {
        "x": np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22.5, 24.5, 27.5, 32, 42, 57, 72]),
        "y": np.array([1.1, 3.4, 8.7, 14.5, 22.5, 28, 33.7, 33.4, 34, 33, 28.4, 24.9, 20.8, 16.4, 10.4, 7.3, 1.2]),
        "title": "Full Dataset"
    },
    {
        "x": np.array([12, 13, 14, 15, 16, 17, 18, 19, 20]),
        "y": np.array([1.1, 3.4, 8.7, 14.5, 22.5, 28, 33.7, 33.4, 34]),
        "title": "First Subset"
    },
    {
        "x": np.array([21, 22.5, 24.5, 27.5, 32, 42, 57, 72]),
        "y": np.array([33, 28.4, 24.9, 20.8, 16.4, 10.4, 7.3, 1.2]),
        "title": "Second Subset"
    }
]

# Plot each data set with a fitted line
plt.figure(figsize=(15, 15))
for i, data in enumerate(data_sets, start=1):
    x = data["x"]
    y = data["y"]
    
    # Construct the design matrix
    A = np.vstack((np.ones_like(x), x)).T

    # Solve the normal equation for coefficients
    coefficients = np.linalg.inv(A.T @ A) @ A.T @ y
    intercept, slope = coefficients

    # Predict y values using the fitted model
    y_predicted = A @ coefficients

    # Plot the data and regression line
    plt.subplot(3, 1, i)
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_predicted, color='red', label=f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}')
    plt.title(data["title"], fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Marijuana Use', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

plt.tight_layout()
plt.show()

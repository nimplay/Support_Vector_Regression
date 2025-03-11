import pandas as pd
import numpy as np

# Configuration
np.random.seed(42)
num_samples = 1000

# Generate synthetic data
experience = np.random.randint(0, 30, num_samples)
educational_level = np.random.choice([1, 2, 3], num_samples)
skills = np.random.randint(1, 10, num_samples)

# Calculating the wage based on a synthetic formula
salary = (
    30 * experience +
    15 * educational_level * experience +
    5 * skills * experience +
    np.random.normal(0, 50, num_samples)
)

# Create a DataFrame
data = pd.DataFrame({
    'Experience': experience,
    'educational_level': educational_level,
    'skills': skills,
    'salary': salary
})

# Saving the DataFrame in a CSV file
data.to_csv('professional_data.csv', index=False)

print("CSV file successfully generated: 'professional_data.csv'")

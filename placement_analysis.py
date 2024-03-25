import pandas as pd
import numpy as np

# Generate synthetic data for the sample dataset
np.random.seed(0)  # for reproducibility

# Generating student IDs
num_students = 1000
student_ids = np.arange(1, num_students + 1)

# Generating student names
student_names = ['Student ' + str(i) for i in student_ids]

# Generating student GPAs (assuming a normal distribution)
mean_gpa = 3.5
std_dev_gpa = 0.5
student_gpas = np.random.normal(mean_gpa, std_dev_gpa, num_students)

# Generating student majors (mock data)
majors = ['Computer Science', 'Electrical Engineering', 'Mechanical Engineering', 'Physics', 'Mathematics']
student_majors = np.random.choice(majors, num_students)

# Generating company names (mock data)
companies = ['Google', 'Microsoft', 'Amazon', 'Apple', 'Tesla', 'Facebook']
placement_companies = np.random.choice(companies, num_students)

# Generating placement status (placed or not placed)
placement_status = np.random.choice(['Placed', 'Not Placed'], num_students, p=[0.8, 0.2])

# Generating salary offered for placed students (assuming a normal distribution)
mean_salary = 80000
std_dev_salary = 10000
salary_offered = np.where(placement_status == 'Placed', np.random.normal(mean_salary, std_dev_salary, num_students), np.nan)

# Create the DataFrame
data = {
    'Student ID': student_ids,
    'Student Name': student_names,
    'GPA': student_gpas,
    'Major': student_majors,
    'Company': placement_companies,
    'Placement Status': placement_status,
    'Salary Offered': salary_offered
}

df = pd.DataFrame(data)


# Save the dataset to a CSV file
df.to_csv('placement_dataset.csv', index=False)

# Load the dataset from the CSV file
df = pd.read_csv('placement_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Basic statistics summary
print(df.describe())

# Count of placed and not placed students
print(df['Placement Status'].value_counts())

# Average salary offered
average_salary = df[df['Placement Status'] == 'Placed']['Salary Offered'].mean()
print('Average Salary Offered: $', round(average_salary, 2))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Read the data from the CSV file
df = pd.read_csv('placement_dataset.csv')

# Perform one-hot encoding for the 'Major' column
encoder = OneHotEncoder()
major_encoded = encoder.fit_transform(df[['Major']])

# Concatenate the encoded 'Major' column with other features
X = pd.concat([df[['GPA']], pd.DataFrame(major_encoded.toarray(), columns=encoder.get_feature_names_out(['Major']))], axis=1)

# Define the target variable
y = df['Placement Status']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", "{:.2f}%".format(accuracy * 100))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

# Read the data from the CSV file
df = pd.read_csv('placement_dataset.csv')

# Perform one-hot encoding for the 'Major' column
encoder = OneHotEncoder()
major_encoded = encoder.fit_transform(df[['Major']])

# Concatenate the encoded 'Major' column with other features
X = pd.concat([df[['GPA']], pd.DataFrame(major_encoded.toarray(), columns=encoder.get_feature_names_out(['Major']))], axis=1)

# Define the target variable
y = df['Placement Status']

# Perform one-hot encoding for the target variable
y = pd.get_dummies(y)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer with 64 neurons and ReLU activation
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons and ReLU activation
model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer with softmax activation for multi-class classification

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", "{:.2f}%".format(accuracy * 100))

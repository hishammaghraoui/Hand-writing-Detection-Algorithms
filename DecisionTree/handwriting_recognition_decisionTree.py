from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load CSV files
train_df = pd.read_csv('tps/emnist-letters-train.csv')
test_df = pd.read_csv("tps/emnist-letters-test.csv")

# Function to preprocess images
def preprocess_images(df):
    # Reshape pixel values to image dimensions
    images = df.iloc[:, 1:].values.reshape(-1, 28, 28)
    # Convert images to grayscale if necessary
    gray_images = []
    for img in images:
        if img.ndim == 3:  # Check if image has three channels (BGR)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img  # Grayscale image already
        gray_images.append(gray_img)
    gray_images = np.array(gray_images)
    # Resize images to a uniform size (28x28)
    resized_images = np.array([cv2.resize(img, (28, 28)) for img in gray_images])
    # Normalize pixel values to the range [0, 1]
    normalized_images = resized_images / 255.0
    return normalized_images


# Preprocess training and testing images
X_train = preprocess_images(train_df)
X_test = preprocess_images(test_df)

# Extract labels
y_train = train_df.iloc[:, 0].values  
y_test = test_df.iloc[:, 0].values 

# Display the shape of preprocessed images
print("Shape of preprocessed training images:", X_train.shape)
print("Shape of preprocessed testing images:", X_test.shape)


# Reshape preprocessed images to a flat feature vector
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Display the shape of the feature vectors
print("Shape of feature vectors for training:", X_train_flat.shape)
print("Shape of feature vectors for testing:", X_test_flat.shape)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train_flat, y_train)

# Make predictions on the testing data
predictions = clf.predict(X_test_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Generate a classification report with zero_division=1
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=1))
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}


# Reduce the Dataset Size
X_train_small, _, y_train_small, _ = train_test_split(X_train_flat, y_train, train_size=0.1, random_state=42)

# Shuffle the smaller dataset
X_train_small, y_train_small = shuffle(X_train_small, y_train_small, random_state=42)

# Use Randomized Search with Parallelization
random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train_small, y_train_small)

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = random_search.best_estimator_

# Evaluate the best model on the testing data
best_predictions = best_model.predict(X_test_flat)
accuracy = accuracy_score(y_test, best_predictions)
print("Accuracy:", accuracy)

# Generate a classification report
print("\nClassification Report:")
print(classification_report(y_test, best_predictions, zero_division=1))




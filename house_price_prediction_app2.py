import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Housing.csv')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# Preprocess the data
def preprocess_data(df):
    df = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 
                                      'hotwaterheating', 'airconditioning', 
                                      'prefarea', 'furnishingstatus'], drop_first=True)

    low_threshold = df['price'].quantile(0.33)
    high_threshold = df['price'].quantile(0.66)
    df['price_category'] = pd.cut(df['price'], bins=[-float('inf'), low_threshold, high_threshold, float('inf')],
                                  labels=['Low', 'Medium', 'High'])
    return df

# Main app
st.title("House Price Prediction App")

# Load and preprocess data
df = load_data()
df = preprocess_data(df)

# Prepare the model for linear regression
X = df.drop(columns=['price', 'price_category'])
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set for evaluation
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
#st.write(f"Linear Regression Model MSE: {mse:.2f}")

#model = LinearRegression()
#model.fit(X, y)

# Input fields for linear regression
st.header("Predict House Price (Linear Regression)")
area = st.number_input("Area (sq ft)", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0)
stories = st.number_input("Number of Stories", min_value=0)
parking = st.number_input("Parking Spaces", min_value=0)

# Prepare the input data for linear regression
if st.button("Predict Price (Linear Regression)"):
    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking
    }

    input_df = pd.DataFrame(input_data, index=[0])

    # Add default values for categorical features
    for col in ['mainroad', 'guestroom', 'basement', 
                'hotwaterheating', 'airconditioning', 
                'prefarea', 'furnishingstatus']:
        input_df[col] = 0  # Default value for all categorical features

    # One-hot encoding
    input_df = pd.get_dummies(input_df, columns=['mainroad', 'guestroom', 'basement', 
                                                  'hotwaterheating', 'airconditioning', 
                                                  'prefarea', 'furnishingstatus'], drop_first=True)

    # Align the input DataFrame with the model's training data columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Save input_df to session state
    st.session_state.input_df = input_df

    # Predict the price
    prediction = model.predict(input_df)
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

# Prepare decision tree model for price category prediction
y_category = df['price_category']

# Split the data into training and testing sets for classification
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_category, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train_cat, y_train_cat)

# Predict on the test set for evaluation
y_pred_cat = decision_tree_model.predict(X_test_cat)

# Calculate the accuracy score to evaluate the classifier
accuracy = accuracy_score(y_test_cat, y_pred_cat)
st.write(f"Decision Tree Classifier Accuracy: {accuracy:.2%}")

#decision_tree_model = DecisionTreeClassifier()
#decision_tree_model.fit(X, y_category)

# Button to predict price category using decision tree
if st.button("Predict Price Category (Decision Tree)"):
    # Ensure input_df is accessed from session state
    if 'input_df' in st.session_state:
        category_prediction = decision_tree_model.predict(st.session_state.input_df)
        st.write(f"Predicted Price Category: {category_prediction[0]}")
    else:
        st.write("Please enter details for linear regression first.")
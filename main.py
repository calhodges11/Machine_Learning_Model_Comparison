import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Set Streamlit page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Title
st.title("🏡 House Price Prediction Model Comparison")

# Dataset selection
dataset = st.selectbox("Select dataset", ["Iowa", "California"])

@st.cache_data
def load_data(name):
    if name == "Iowa":
        data = pd.read_csv("train.csv")
        X = data[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
        y = data['SalePrice']
    else:
        data = pd.read_csv("housing.csv")
        data_encoded = pd.get_dummies(data, columns=['ocean_proximity'])
        X = data_encoded.drop('median_house_value', axis=1)
        y = data_encoded['median_house_value']
    return X, y

# Load and prepare data
X, y = load_data(dataset)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
imputer = SimpleImputer(strategy='mean')
train_X = imputer.fit_transform(train_X)
val_X = imputer.transform(val_X)
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)

# Model selection
model_choices = st.multiselect(
    "Select models to evaluate",
    ["Random Forest", "Lasso Regression", "Neural Network"],
    default=["Random Forest", "Lasso Regression", "Neural Network"]
)

results = []
predictions = {}

# Random Forest
if "Random Forest" in model_choices:
    start = time.time()
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    mae = mean_absolute_error(val_y, pred)
    duration = time.time() - start
    results.append(("Random Forest", mae, duration))
    predictions["Random Forest"] = pred

# Lasso Regression
if "Lasso Regression" in model_choices:
    start = time.time()
    model = Lasso(alpha=3.0)
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    mae = mean_absolute_error(val_y, pred)
    duration = time.time() - start
    results.append(("Lasso Regression", mae, duration))
    predictions["Lasso Regression"] = pred

# Neural Network
if "Neural Network" in model_choices:
    start = time.time()
    nn_model = Sequential()
    nn_model.add(Dense(100, input_shape=(train_X.shape[1],), activation='relu'))
    nn_model.add(Dense(1000, activation='relu'))
    nn_model.add(Dense(1, activation='linear'))
    nn_model.compile(loss='mae', optimizer='adam')
    nn_model.fit(train_X, train_y, epochs=10, batch_size=32, verbose=0, validation_data=(val_X, val_y))
    pred = nn_model.predict(val_X).flatten()
    mae = mean_absolute_error(val_y, pred)
    duration = time.time() - start
    results.append(("Neural Network", mae, duration))
    predictions["Neural Network"] = pred

# Show results
if results:
    st.subheader("📈 Model Results")
    results_df = pd.DataFrame(results, columns=["Model", "MAE", "Time (s)"])
    st.dataframe(results_df.style.format({"MAE": "{:.0f}", "Time (s)": "{:.2f}"}), use_container_width=True)

    # Plot predictions
    st.subheader("📊 Prediction Scatter Plot")
    fig, ax = plt.subplots()
    for name, pred in predictions.items():
        ax.scatter(val_y, pred, label=name, alpha=0.6)
    ax.plot([val_y.min(), val_y.max()], [val_y.min(), val_y.max()], 'k--', label="Ideal")
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Please select at least one model to run.")

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
import numpy as np
# Optional TensorFlow support
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    tensorflow_available = True
except ImportError:
    tensorflow_available = False

st.set_page_config(page_title="🏡 House Price Predictor", layout="wide")
st.title("🏡 House Price Prediction Model Comparison")

# ------------------- Data Upload & Selection -------------------
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📌 Select target and feature columns")
    target_col = st.selectbox("Select the target column (what you're predicting)", data.columns)
    feature_cols = st.multiselect("Select feature columns (inputs for the model)",
                                  [col for col in data.columns if col != target_col])

    if not feature_cols:
        st.warning("Please select at least one feature column to continue.")
        st.stop()

    X = data[feature_cols]
    y = data[target_col]

else:
    st.subheader("📂 Or choose a built-in dataset")
    dataset = st.selectbox("Select dataset", ["Iowa", "California"])


    @st.cache_data
    def load_data(name):
        if name == "Iowa":
            df = pd.read_csv("train.csv")
            X = df[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
            y = df['SalePrice']
        else:
            df = pd.read_csv("housing.csv")
            df = pd.get_dummies(df, columns=['ocean_proximity'])
            X = df.drop('median_house_value', axis=1)
            y = df['median_house_value']
        return X, y


    X, y = load_data(dataset)

# ------------------- Preprocessing -------------------
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
imputer = SimpleImputer(strategy='mean')
train_X = imputer.fit_transform(train_X)
val_X = imputer.transform(val_X)
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)

# ------------------- Model Selection -------------------
available_models = ["Random Forest", "Lasso Regression"]
if tensorflow_available:
    available_models.append("Neural Network")

model_choices = st.multiselect(
    "Select models to evaluate",
    available_models,
    default=available_models
)

results = []
predictions = {}

if "Random Forest" in model_choices:
    start = time.time()
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    pred = np.maximum(pred, 0)
    mae = mean_absolute_error(val_y, pred)
    duration = time.time() - start
    results.append(("Random Forest", mae, duration))
    predictions["Random Forest"] = pred

if "Lasso Regression" in model_choices:
    start = time.time()
    model = Lasso(alpha=3.0)
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    pred = np.maximum(pred, 0)
    mae = mean_absolute_error(val_y, pred)
    duration = time.time() - start
    results.append(("Lasso Regression", mae, duration))
    predictions["Lasso Regression"] = pred

if "Neural Network" in model_choices and tensorflow_available:
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

if not tensorflow_available:
    st.info("💡 TensorFlow not installed. Neural Network option is unavailable.")

# ------------------- Results -------------------
if results:
    st.subheader("📈 Model Results")
    results_df = pd.DataFrame(results, columns=["Model", "MAE", "Time (s)"])
    st.dataframe(results_df.style.format({"MAE": "{:.0f}", "Time (s)": "{:.2f}"}), use_container_width=True)

    st.subheader("📊 Prediction Scatter Plot")
    fig, ax = plt.subplots()
    for name, pred in predictions.items():
        ax.scatter(val_y, pred, label=name, alpha=0.6)
    ax.plot([val_y.min(), val_y.max()], [val_y.min(), val_y.max()], 'k--', label="Ideal")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Please select at least one model to run.")

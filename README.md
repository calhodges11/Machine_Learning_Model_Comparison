----- Machine Learning Model Comparison Tool -----

This is a streamlit application for comparing lasso regression, random forest, and neural network models. 
It compares the performance of these models when tasked with predicting housing prices using their qualities.
Users can upload their own CSV or use one of two built in datasets. 

To run the app:
1. Clone the repository
2. Create a virtual environment with python -m venv streamlit-env
3. Activate it with streamlit-env\Scripts\activate
4. Install dependencies with pip install -r requirements.txt
5. Run the app with streamlit run main.py


NOTE: Tensorflow is not yet compatible with Python 3.12 and up, if tensorflow is not installed the option for neural network will not be available. 

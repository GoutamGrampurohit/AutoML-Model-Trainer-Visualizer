import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Advanced AutoML Trainer", layout="wide")

st.title("🚀 Advanced AutoML Trainer")
st.write("Upload your data, and this app will automatically handle missing values, encode features, and train a model.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    # Initialize session state for suggestion
    if 'suggested_task_index' not in st.session_state:
        st.session_state.suggested_task_index = 0

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        target_column = st.selectbox("Select the Target Column", df.columns)

        # --- Problem 4: Target Column Suitability ---
        if target_column:
            target_series = df[target_column]
            # Check if target is numeric and has high cardinality for regression
            if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
                st.session_state.suggested_task_index = 1 # Suggest Regression
            else:
                st.session_state.suggested_task_index = 0 # Suggest Classification
        
        # User selects the task, with a smart default
        model_type = st.radio(
            "Select Task", 
            ('Classification', 'Regression'), 
            index=st.session_state.suggested_task_index
        )
        
        # Display a warning if the user's choice mismatches the suggestion
        if (model_type == 'Classification' and st.session_state.suggested_task_index == 1) or \
           (model_type == 'Regression' and st.session_state.suggested_task_index == 0):
            st.warning("Warning: The selected task may not be suitable for your target variable.")

        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100.0

# Main panel
if df is not None:
    st.header("Data Preview")
    st.dataframe(df.head())
    
    if model_type == 'Classification':
        model_name = st.sidebar.selectbox("Select Model", ('Logistic Regression', 'Random Forest Classifier'))
    else:
        model_name = st.sidebar.selectbox("Select Model", ('Linear Regression', 'Random Forest Regressor'))
    
    train_button = st.sidebar.button("Train Model", type="primary")
else:
    st.info("Awaiting for CSV file to be uploaded.")

# --- Model Training and Evaluation ---
if 'train_button' in locals() and train_button:
    st.header("📊 Results")

    # 1. Prepare Data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # --- Problems 2 & 3: Handle Missing Values and Non-Numeric Features ---
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    st.write("Identified **{} numeric** and **{} categorical** features.".format(len(numeric_features), len(categorical_features)))

    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')) # Fill missing numerics with mean
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing categoricals with mode
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Convert categories to numeric
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # 2. Define Model
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest Classifier': RandomForestClassifier(),
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor()
    }
    model = models[model_name]

    # 3. Create the full pipeline
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)])

    # 4. Split data and train the pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    # 5. Display Metrics and Visuals
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        if model_type == 'Classification':
            accuracy = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred, average='weighted')
            st.write(f"**Accuracy:** `{accuracy:.4f}`")
            st.write(f"**F1-Score:** `{f1:.4f}`")

            # Confusion Matrix
            cm = metrics.confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                        xticklabels=full_pipeline.classes_, yticklabels=full_pipeline.classes_)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
        else: # Regression
            mae = metrics.mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            r2 = metrics.r2_score(y_test, y_pred)
            st.write(f"**Mean Absolute Error (MAE):** `{mae:.4f}`")
            st.write(f"**Root Mean Squared Error (RMSE):** `{rmse:.4f}`")
            st.write(f"**R-squared ($R^2$):** `{r2:.4f}`")

            # Scatter Plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            st.pyplot(fig)
            
    with col2:
        st.subheader("Predictions on Test Set")
        # Reset index to align for concatenation
        y_test_reset = y_test.reset_index(drop=True)
        predictions_df = pd.DataFrame({'Actual': y_test_reset, 'Predicted': y_pred})
        st.dataframe(predictions_df)
        
        # Cache the conversion to prevent re-running on every interaction
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(predictions_df)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f'{model_name}_predictions.csv',
            mime='text/csv',
        )
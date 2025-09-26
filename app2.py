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
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="AutoML Trainer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.step-container {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    text-align: center;
    margin: 0.5rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-card h2 {
    color: white !important;
    font-size: 2.5rem !important;
    margin: 0.5rem 0 !important;
}
.metric-card h3 {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1rem !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header"><h1>🚀 AutoML Trainer</h1><p>Upload your data and train machine learning models automatically</p></div>', unsafe_allow_html=True)

# --- Step 1: File Upload ---
st.markdown('<div class="step-container">', unsafe_allow_html=True)
st.header("📁 Step 1: Upload Your Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type=["csv"],
    help="Upload a CSV file with your training data"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Data preview
        with st.expander("🔍 Data Preview", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            with col2:
                st.subheader("Dataset Info")
                st.write(f"**Rows:** {df.shape[0]:,}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Missing values:** {df.isnull().sum().sum()}")
                st.write(f"**Numeric columns:** {len(df.select_dtypes(include=np.number).columns)}")
                st.write(f"**Categorical columns:** {len(df.select_dtypes(include=['object', 'category']).columns)}")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()
else:
    st.info("👆 Please upload a CSV file to get started")
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# --- Step 2: Configure Training ---
st.markdown('<div class="step-container">', unsafe_allow_html=True)
st.header("⚙️ Step 2: Configure Your Model")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if df.empty:
        st.error("No data available")
        st.stop()
    
    target_column = st.selectbox(
        "🎯 Select Target Column", 
        df.columns,
        help="Choose the column you want to predict"
    )

with col2:
    # Smart task suggestion
    if target_column:
        target_series = df[target_column]
        
        # Check for task type
        if pd.api.types.is_numeric_dtype(target_series):
            unique_values = target_series.nunique()
            if unique_values > 15:
                suggested_task = "Regression"
                task_icon = "📈"
            else:
                suggested_task = "Classification"  
                task_icon = "🏷️"
        else:
            suggested_task = "Classification"
            task_icon = "🏷️"
        
        st.info(f"{task_icon} Suggested: **{suggested_task}**")
        
        model_type = st.radio(
            "📊 Task Type",
            ['Classification', 'Regression'],
            index=0 if suggested_task == 'Classification' else 1
        )

with col3:
    test_size = st.slider(
        "🔄 Test Set Size (%)", 
        min_value=10, 
        max_value=50, 
        value=20,
        help="Percentage of data to use for testing"
    ) / 100.0

# Model selection
if model_type == 'Classification':
    model_options = {
        'Logistic Regression': '📊',
        'Random Forest Classifier': '🌲'
    }
else:
    model_options = {
        'Linear Regression': '📈', 
        'Random Forest Regressor': '🌲'
    }

model_name = st.selectbox(
    "🤖 Select Model",
    list(model_options.keys()),
    format_func=lambda x: f"{model_options[x]} {x}"
)

# Validation
if target_column:
    # Check if target has missing values
    missing_target = df[target_column].isnull().sum()
    if missing_target > 0:
        st.warning(f"⚠️ Target column has {missing_target} missing values. These rows will be removed.")
    
    # Check if enough data after removing missing targets
    valid_rows = df[target_column].notna().sum()
    if valid_rows < 10:
        st.error("❌ Not enough valid data points for training (minimum 10 required)")
        st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# --- Step 3: Train Model ---
st.markdown('<div class="step-container">', unsafe_allow_html=True)
st.header("🚀 Step 3: Train Your Model")

train_button = st.button("🏃‍♂️ Train Model", type="primary", use_container_width=True)

if train_button:
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Data preparation
        status_text.text("🔄 Preparing data...")
        progress_bar.progress(20)
        
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_column]).copy()
        
        if df_clean.empty:
            st.error("❌ No valid data remaining after removing missing target values")
            st.stop()
        
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Feature identification
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        status_text.text("🛠️ Building preprocessing pipeline...")
        progress_bar.progress(40)
        
        # Create preprocessing pipeline with scaling
        from sklearn.preprocessing import StandardScaler
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())  # Add scaling for better performance
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Model selection with better hyperparameters
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=2000, 
                random_state=42, 
                C=1.0, 
                class_weight='balanced' if model_type == 'Classification' else None
            ),
            'Random Forest Classifier': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        model = models[model_name]
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        status_text.text("🎯 Training model...")
        progress_bar.progress(60)
        
        # Train-test split with stratification for classification
        if model_type == 'Classification' and len(y.unique()) > 1:
            # For classification with multiple classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # For regression or single class
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Additional data quality check
        st.info(f"📊 Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Data quality insights before training
        st.info(f"🔍 **Data Quality Check:**")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.write(f"**Class Distribution (Top 5):**")
            if model_type == 'Classification':
                class_dist = y_train.value_counts().head()
                for class_val, count in class_dist.items():
                    st.write(f"• {class_val}: {count} ({count/len(y_train)*100:.1f}%)")
            else:
                st.write(f"Target range: {y_train.min():.2f} to {y_train.max():.2f}")
                st.write(f"Mean: {y_train.mean():.2f}")
                st.write(f"Std: {y_train.std():.2f}")
        
        with col_info2:
            st.write(f"**Features:**")
            st.write(f"• Numeric: {len(numeric_features)}")
            st.write(f"• Categorical: {len(categorical_features)}")
            total_missing = X_train.isnull().sum().sum()
            st.write(f"• Missing values: {total_missing}")
            
        with col_info3:
            st.write(f"**Recommendations:**")
            if model_type == 'Classification':
                min_class_size = y_train.value_counts().min()
                if min_class_size < 10:
                    st.warning("⚠️ Small classes detected")
                elif len(y.unique()) > 10:
                    st.warning("⚠️ Many classes (>10)")
                else:
                    st.success("✅ Good class balance")
            if len(categorical_features) > len(numeric_features) * 2:
                st.info("💡 Many categorical features - consider feature selection")
        
        # Train model
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)
        
        status_text.text("📊 Generating results...")
        progress_bar.progress(80)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Results ---
        st.header("📈 Results")
        
        # Performance metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        if model_type == 'Classification':
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            with metrics_col1:
                st.markdown(f'<div class="metric-card"><h3>🎯 Accuracy</h3><h2>{accuracy:.3f}</h2></div>', unsafe_allow_html=True)
            with metrics_col2:
                st.markdown(f'<div class="metric-card"><h3>🎪 Precision</h3><h2>{precision:.3f}</h2></div>', unsafe_allow_html=True)
            with metrics_col3:
                st.markdown(f'<div class="metric-card"><h3>🔍 Recall</h3><h2>{recall:.3f}</h2></div>', unsafe_allow_html=True)
        
        else:  # Regression
            mae = metrics.mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            r2 = metrics.r2_score(y_test, y_pred)
            
            with metrics_col1:
                st.markdown(f'<div class="metric-card"><h3>📏 MAE</h3><h2>{mae:.3f}</h2></div>', unsafe_allow_html=True)
            with metrics_col2:
                st.markdown(f'<div class="metric-card"><h3>📐 RMSE</h3><h2>{rmse:.3f}</h2></div>', unsafe_allow_html=True)
            with metrics_col3:
                st.markdown(f'<div class="metric-card"><h3>📊 R²</h3><h2>{r2:.3f}</h2></div>', unsafe_allow_html=True)
        
        # Visualizations and predictions
        viz_col, pred_col = st.columns([3, 2])
        
        with viz_col:
            st.subheader("📊 Model Performance")
            
            # Show class distribution for classification problems
            if model_type == 'Classification':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Confusion Matrix
                cm = metrics.confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax1)
                ax1.set_xlabel('Predicted Labels', fontsize=12)
                ax1.set_ylabel('True Labels', fontsize=12)
                ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                
                # Class distribution comparison
                train_dist = y_train.value_counts().sort_index()
                test_dist = y_test.value_counts().reindex(train_dist.index, fill_value=0)
                
                x = np.arange(len(train_dist))
                width = 0.35
                
                ax2.bar(x - width/2, train_dist.values, width, label='Training', alpha=0.8, color='#667eea')
                ax2.bar(x + width/2, test_dist.values, width, label='Test', alpha=0.8, color='#764ba2')
                
                ax2.set_xlabel('Classes', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(train_dist.index, rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
            else:  # Regression
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Actual vs Predicted scatter plot
                ax1.scatter(y_test, y_pred, alpha=0.6, color='#667eea', s=50)
                
                # Perfect prediction line
                min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                ax1.set_xlabel('Actual Values', fontsize=12)
                ax1.set_ylabel('Predicted Values', fontsize=12)
                ax1.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Residual plot
                residuals = y_test - y_pred
                ax2.scatter(y_pred, residuals, alpha=0.6, color='#764ba2', s=50)
                ax2.axhline(y=0, color='r', linestyle='--', lw=2)
                ax2.set_xlabel('Predicted Values', fontsize=12)
                ax2.set_ylabel('Residuals', fontsize=12)
                ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with pred_col:
            st.subheader("🔮 Predictions Sample")
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'Actual': y_test.reset_index(drop=True),
                'Predicted': y_pred
            })
            
            if model_type == 'Regression':
                predictions_df['Error'] = abs(predictions_df['Actual'] - predictions_df['Predicted'])
            
            # Show first 15 predictions
            st.dataframe(predictions_df.head(15), use_container_width=True)
            
            # Download button
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Predictions",
                data=csv,
                file_name=f'{model_name.lower().replace(" ", "_")}_predictions.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        # Feature importance (for tree-based models)
        if 'Random Forest' in model_name:
            st.subheader("🎯 Feature Importance")
            
            try:
                # Get feature names after preprocessing
                feature_names = (numeric_features + 
                               list(full_pipeline.named_steps['preprocessor']
                                   .named_transformers_['cat']
                                   .named_steps['onehot']
                                   .get_feature_names_out(categorical_features)))
                
                importances = full_pipeline.named_steps['model'].feature_importances_
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis')
                ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
                ax.set_xlabel('Importance Score', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.info("Feature importance visualization not available for this model configuration")
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed successfully!")
        st.success(f"🎉 Model trained successfully! Test accuracy/R²: {(accuracy if model_type == 'Classification' else r2):.3f}")
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.error("Please check your data and try again. Make sure your target column is appropriate for the selected task type.")

st.markdown('</div>', unsafe_allow_html=True)

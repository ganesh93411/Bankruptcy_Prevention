import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

# Add custom CSS to style the top border and text
st.markdown(
    """
    <style>
    .top-border {
        background-color: purple;   /* Border color */
        color: white;  /* Text color */
        font-size: 30px;  /* Text size - increase for larger text */
        font-family: 'Arial', sans-serif;  /* Font style - change to your preferred font */
        font-weight: bold;  /* Text weight */
        padding: 20px 0;  /* Increase padding to make the border taller */
        text-align: center;  /* Center the text horizontally */
        width: 100%;  /* Full width of the page */
    }
    </style>
    """, unsafe_allow_html=True
)

# Add the top border with text
st.markdown('<div class="top-border"> Bankruptcy Prediction App </div>', unsafe_allow_html=True)

# title
st.title(" Bankruptcy Prevention")

# Display the image at the top with a fixed height and width (rectangular shape)
st.image("https://www.durrettebradshaw.com/wp-content/uploads/2018/08/Bankruptcy.jpg", width=800)

st.write("This application explores the Bankruptcy Dataset And Uses Machine Learning Models for Prediction.")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = {}
if 'model_performance' not in st.session_state:
    st.session_state['model_performance'] = {}

# Upload Dataset
uploaded_file = st.file_uploader("Upload the Bankruptcy Dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
        # Load the dataset
        data = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write("Dataset Loaded Successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(data)  # Display the DataFrame

        # Shape of Data
        st.write("Dataset Shape:")
        st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

        # Ensure it is the Bankruptcy dataset by checking specific columns
        required_columns = {'industrial_risk', 'management_risk', 'financial_flexibility','credibility', 'competitiveness', 'operating_risk', 'class'}
        if not required_columns.issubset(data.columns):
           st.error("This is not the Bankruptcy dataset! Please upload the correct file.")
        else:
            # Data Preprocessing
            st.subheader("Data Preprocessing")

            # Drop Duplicates
            if data.duplicated().sum() > 0:
                st.write("Duplicate rows found. Dropping duplicates...")
                data = data.drop_duplicates()
            st.write(f"Dataset shape after dropping duplicates: {data.shape}")

            # Summary Statistics
            if st.checkbox("Show Summary Statistics"):
                 st.markdown('<h4 style="color: navy;"> Summary Statistics: </h4>', unsafe_allow_html=True)
                 st.write(data.describe())

            # Renaming the dataset  
            cleaned_data = data.iloc[:,:]

            # Assinging featues and target variables 

            features = data.drop(columns = ['class'])
            target = data['class']
            
            # Splitting the data into features and target variables 
            X = cleaned_data.drop(columns=['class']) # target_column is class
            y = cleaned_data['class']
          
            # Visualizations
            if st.checkbox("Show Visualizations"):
                 st.markdown('<h4 style="color: navy;">Numerical Feature Distribution: </h4>', unsafe_allow_html=True)
                 numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                 for column in numeric_columns:
                     fig, ax = plt.subplots()
                     sns.histplot(data[column], kde=True, ax=ax)
                     st.write(f"Distribution of {column}")
                     st.pyplot(fig)

                 st.markdown('<h4 style="color: navy;">Categorical Feature Count: </h4>', unsafe_allow_html=True)
                 categorical_columns = data.select_dtypes(include=['object']).columns
                 for column in categorical_columns:
                      fig, ax = plt.subplots()
                      sns.countplot(data=data, x=column, palette = "turbo")
                      st.write(f"Count Plot for {column}")
                      st.pyplot(fig)

                 st.markdown('<h4 style="color: navy;">Correlation Heatmap: </h4>', unsafe_allow_html=True)
                 if len(numeric_columns) > 1:
                      fig, ax = plt.subplots()
                      sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
                      st.pyplot(fig)

                 # Scatter Plot with Relationship Insights
                 st.markdown('<h4 style="color: navy;">Scatter Plot: </h4>', unsafe_allow_html=True)
                 x_axis = st.selectbox("Select X-axis for Scatter Plot", options=numeric_columns)
                 y_axis = st.selectbox("Select Y-axis for Scatter Plot", options=numeric_columns)

                 if x_axis and y_axis:
                      fig, ax = plt.subplots()
                      sns.scatterplot(x=X[x_axis], y=X[y_axis], ax=ax)
                      ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
                      st.pyplot(fig)

                      # Calculate and display correlation
                      correlation = np.corrcoef(X[x_axis], X[y_axis])[0, 1]
                      st.write(f"Correlation between {x_axis} and {y_axis}: **{correlation:.2f}**")

                      # Provide understanding of the relationship
                      if correlation > 0.7:
                           st.info(f"There is a **strong positive relationship** between {x_axis} and {y_axis}. As {x_axis} increases, {y_axis} tends to increase.")
                      elif 0.3 < correlation <= 0.7:
                           st.info(f"There is a **moderate positive relationship** between {x_axis} and {y_axis}. {x_axis} and {y_axis} are somewhat correlated.")
                      elif 0.1 < correlation <= 0.3:
                           st.info(f"There is a **weak positive relationship** between {x_axis} and {y_axis}. {x_axis} and {y_axis} have minimal correlation.")
                      elif -0.1 <= correlation <= 0.1:
                           st.info(f"There is **no significant relationship** between {x_axis} and {y_axis}.")
                      elif -0.3 <= correlation < -0.1:
                           st.info(f"There is a **weak negative relationship** between {x_axis} and {y_axis}. {y_axis} tends to decrease slightly as {x_axis} increases.")
                      elif -0.7 <= correlation < -0.3:
                           st.info(f"There is a **moderate negative relationship** between {x_axis} and {y_axis}. As {x_axis} increases, {y_axis} tends to decrease.")
                      else:
                           st.info(f"There is a **strong negative relationship** between {x_axis} and {y_axis}. As {x_axis} increases, {y_axis} significantly decreases.")

            # Ensure target variable is binary
            if y.dtype == "object":
                 y = y.map({"bankruptcy": 1, "non-bankruptcy": 0})

            # Standardizing numeric features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Model Selection and Training
            st.header("Model Building")
            model_choice = st.selectbox("Select Model for Training", ("Logistic Regression", "KNN", "Random Forest", "SVM"))

            if model_choice == "Logistic Regression":
                 # Hyperparameter for Logistic Regression
                 penalty = st.selectbox("Penalty", options=["l2", "l1", "elasticnet", "none"])
                 solver = "saga" if penalty in ["l1", "elasticnet"] else "lbfgs"
                 C = st.slider("Inverse Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)

                 # Include l1_ratio for elasticnet
                 if penalty == "elasticnet":
                      l1_ratio = st.slider("L1 Ratio (ElasticNet)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                      model = LogisticRegression(penalty=penalty, C=C, solver=solver, l1_ratio=l1_ratio, max_iter=1000)
                 elif penalty == "none":
                      model = LogisticRegression(penalty=None, C=C, solver=solver, max_iter=1000)
                 else:
                      model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)
            elif model_choice == "KNN":
                 n_neighbors = st.slider("Select number of neighbors (for KNN)", min_value=1, max_value=20, value=5, step=1)
                 weights = st.selectbox("Weight Function", options=["uniform", "distance"])
                 model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            elif model_choice == "Random Forest":
                 n_estimators = st.slider("Select number of trees (for Random Forest)", min_value=10, max_value=200, value=100, step=10)
                 max_depth = st.slider("Select max depth (for Random Forest)", min_value=1, max_value=20, value=10, step=1)
                 min_samples_split = st.slider("Minimum Samples Split", min_value=2, max_value=10, value=2, step=1)
                 model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            elif model_choice == "SVM":
                 kernel = st.selectbox("Select kernel (for SVM)", ("linear", "rbf", "poly", "sigmoid"))
                 C = st.slider("Select C value (for SVM)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                 gamma = st.selectbox("Kernel Coefficient (Gamma)", options=["scale", "auto"])
                 model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
            
            st.write("Please train the model to make predictions.")

            if st.button("Train Model"):
                 model.fit(X_train, y_train)
                 st.session_state['model'] = model  # Save the trained model
                 st.session_state['features'] = features  # Save feature names for validation
                 y_pred = model.predict(X_test)
                 y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                 accuracy = accuracy_score(y_test, y_pred)

                 # Save metrics to session state
                 model_metrics = {
                      'accuracy': accuracy,
                      'confusion_matrix': confusion_matrix(y_test, y_pred),
                      'classification_report': classification_report(y_test, y_pred),
                      'precision': precision_score(y_test, y_pred),
                      'recall': recall_score(y_test, y_pred),
                      'f1_score': f1_score(y_test, y_pred),
                      'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A",
                      'roc_curve': roc_curve(y_test, y_proba) if y_proba is not None else None
                 }
                 st.session_state['metrics'] = model_metrics
                 st.session_state['model_performance'][model_choice] = model_metrics  # Store metrics for comparison

                 st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Display Performance Metrics
            if st.session_state['metrics']:
                 if st.checkbox("Show Performance Metrics"):
                      metrics = st.session_state['metrics']
                      st.write(f"### Performance Metrics for {model_choice}")
                      st.write(f"**Precision**: {metrics['precision']:.2f}")
                      st.write(f"**Recall**: {metrics['recall']:.2f}")
                      st.write(f"**F1 Score**: {metrics['f1_score']:.2f}")
                      st.write(f"**ROC AUC Score**: {metrics['roc_auc']}")

                      # Confusion Matrix with model name
                      st.subheader(f"Confusion Matrix for {model_choice}")
                      cm = metrics['confusion_matrix']
                      fig, ax = plt.subplots()
                      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                      st.pyplot(fig)

                      # ROC Curve with model name and AUC Score
                      if metrics['roc_curve'] is not None:
                           fpr, tpr, _ = metrics['roc_curve']
                           roc_auc = metrics['roc_auc']
                           
                           st.subheader(f"ROC Curve for {model_choice}")
                           fig, ax = plt.subplots()
                           ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
                           ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                           ax.set_xlabel("False Positive Rate")
                           ax.set_ylabel("True Positive Rate")
                           ax.set_title(f"ROC Curve for {model_choice}")
                           ax.legend()
                           st.pyplot(fig)
                           
                           # Display AUC Score below the chart
                           st.write(f"**ROC-AUC Score**: {roc_auc:.2f}")

# User Input for Prediction
st.sidebar.header("Make Predictions")
            
if st.session_state.get('model') is not None and st.session_state.get('features') is not None:
     user_input = []
     for feature in st.session_state['features']:
          value = st.sidebar.number_input(f"Enter value for {feature}", value=0.0, step=0.1, format="%.1f", min_value=0.0, max_value=1.0)
          user_input.append(value)

     user_input = np.array(user_input).reshape(1, -1)

     if st.sidebar.button("Predict"):
          prediction = st.session_state['model'].predict(user_input)
          prediction_proba = st.session_state['model'].predict_proba(user_input)[0]
          result_prob = prediction_proba[prediction[0]]  # Extract probability of predicted class
          st.sidebar.write("Prediction Result:", "Bankruptcy" if prediction[0] == 1 else "No Bankruptcy")
          st.sidebar.write("Prediction Probability:", f"{result_prob:.2f}")

# Model Comparison
     if st.checkbox("Compare Models"):
          if st.session_state['model_performance']:
               # Create a performance summary without confusion_matrix and roc_curve for comparison
               comparison_metrics = {
                    model: {
                         key: (value if not isinstance(value, (np.ndarray, tuple)) else "N/A")
                         for key, value in metrics.items()
                    }
                    for model, metrics in st.session_state['model_performance'].items()
               }
               # Convert metrics dictionary to DataFrame
               performance_df = pd.DataFrame(comparison_metrics).T
               performance_df.reset_index(inplace=True)
               performance_df.rename(columns={'index': 'Model'}, inplace=True)

               st.subheader("Model Comparison Table")
               st.write(performance_df)

               st.subheader("Comparison Bar Chart")
               metric_to_compare = st.selectbox("Select Metric for Comparison", ['accuracy', 'precision', 'recall', 'f1_score'])

               fig, ax = plt.subplots()
               sns.barplot(x='Model', y=metric_to_compare, data=performance_df, ax=ax, palette="viridis")
               ax.set_ylabel(metric_to_compare.capitalize())
               ax.set_title(f"Model Comparison: {metric_to_compare.capitalize()}")
               st.pyplot(fig)

               # Identify the best model based on the selected metric
               best_model_row = performance_df.loc[performance_df[metric_to_compare].idxmax()]
               best_model_name = best_model_row['Model']
               best_model_value = best_model_row[metric_to_compare]

               st.success(f"The best-performing model based on **{metric_to_compare}** is **{best_model_name}** with a score of **{best_model_value:.2f}**.")

               # Display ROC AUC scores if available
               roc_scores = {model: metrics.get('roc_auc', "N/A") for model, metrics in st.session_state['model_performance'].items()}
               st.subheader("ROC AUC Scores")
               for model, score in roc_scores.items():
                    st.write(f"{model}: {score}")

    


    

          






                

                

                    
                







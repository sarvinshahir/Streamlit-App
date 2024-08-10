import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, classification_report
import numpy as np

# Custom title with HTML and CSS for styling
st.markdown("""
    <div style="background-color: pink; padding: 10px; border-radius: 10px;">
        <h1 style="color: black; text-align: center;">Dataset Classification App</h1>
        <h3 style="color: black; text-align: center;">CPSC 4820 Final</h3>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for uploading the dataset
st.sidebar.header("Upload Your CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data Integrity Checks Section
    with st.expander("Data Integrity Checks"):
        st.subheader("Data Integrity Checks")
        if st.button("Check for Missing Values"):
            st.write(df.isnull().sum())
        
        if st.button("Display Dataset Shape"):
            st.write(df.shape)
        
        if st.button("Display First Few Rows"):
            st.write(df.head())

        if st.button("Check for Duplicates"):
            st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

        if st.button("Display Data Types"):
            st.write(df.dtypes)

        if st.button("Identify Outliers"):
            # Select only numeric columns for outlier detection
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                st.write("No numeric columns available for outlier detection.")
            else:
                st.write("Outliers based on IQR:")
                Q1 = numeric_df.quantile(0.25)
                Q3 = numeric_df.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
                st.write(outliers)
        
        if st.button("Missing Data Heatmap"):
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
            st.pyplot(plt)
    
    # Exploratory Data Analysis (EDA) Section
    with st.expander("Exploratory Data Analysis (EDA)"):
        st.subheader("Exploratory Data Analysis (EDA)")
        
        if st.button("Generate Histograms"):
            df.hist(bins=20, figsize=(10, 10))
            st.pyplot(plt)
        
        if st.button("Generate Pair Plots"):
            sns.pairplot(df)
            st.pyplot(plt)

        if st.button("Generate Pair Plots with Hue"):
            target_column = st.selectbox("Select the target column for hue", df.columns)
            if target_column:
                sns.pairplot(df, hue=target_column)
                st.pyplot(plt)
        
        if st.button("Generate Correlation Heatmap"):
            # Select only numeric columns for correlation
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                st.write("No numeric columns available for correlation heatmap.")
            else:
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                st.pyplot(plt)
        
        if st.button("Generate Boxplots"):
            df.plot(kind='box', subplots=True, layout=(2,4), figsize=(12,8))
            st.pyplot(plt)
        
        if st.button("Generate Value Counts"):
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            for column in categorical_columns:
                st.write(f"Value counts for {column}:")
                st.write(df[column].value_counts())
        
        if st.button("Generate Distribution Plots"):
            numeric_columns = df.select_dtypes(include([np.number])).columns
            for column in numeric_columns:
                sns.distplot(df[column])
                st.pyplot(plt)
        
        if st.button("Generate Scatter Matrix"):
            pd.plotting.scatter_matrix(df, figsize=(12, 8))
            st.pyplot(plt)

    # Model Selection and Training
    with st.expander("Model Selection and Training"):
        st.subheader("Model Selection and Training")
        
        model_name = st.selectbox("Choose Model", ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors"))
        
        scale_option = st.selectbox("Scaling Option", ("None", "StandardScaler", "MinMaxScaler"))
        
        if st.button("Split Dataset into Training and Testing Sets"):
            X = df.iloc[:, :-1]  # Features
            y = df.iloc[:, -1]   # Target
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            if scale_option == "StandardScaler":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif scale_option == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            
            st.write("Data Split Done")
        
        if 'X_train' in st.session_state and st.button("Train Model"):
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            
            if model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            
            st.write("Model Trained")
    
    # Prediction and Evaluation Section
    with st.expander("Prediction and Evaluation"):
        st.subheader("Prediction and Evaluation")
                
        if 'model' in st.session_state:
            if st.button("Evaluate Model"):
                model = st.session_state['model']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                # Get predictions
                y_pred = model.predict(X_test)
                
                # Calculate overall metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Display overall metrics
                st.write(f"Accuracy: {accuracy}")
                st.write(f"Precision: {precision}")
                st.write(f"Recall: {recall}")
                st.write(f"F1 Score: {f1}")
                
                # Display classification report
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                
                # Display confusion matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)
                
                # Calculate and display class-wise metrics
                precision_per_class = precision_score(y_test, y_pred, average=None)
                recall_per_class = recall_score(y_test, y_pred, average=None)
                f1_per_class = f1_score(y_test, y_pred, average=None)
                st.write(f"Precision per class: {precision_per_class}")
                st.write(f"Recall per class: {recall_per_class}")
                st.write(f"F1 Score per class: {f1_per_class}")

                # Binarize the output for multiclass ROC curve
                y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
                
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)
                else:
                    y_score = model.decision_function(X_test)
        
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(y_test_binarized.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Plot ROC curve for each class
                plt.figure()
                for i in range(len(np.unique(y_test))):
                    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
                
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                st.pyplot(plt)
            
            # Download Predictions with Enhanced Information
            # Download Predictions with Enhanced Information
            if st.button("Predictions"):
                model = st.session_state['model']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                # Ensure X_test and y_test are aligned and check their lengths
                assert len(X_test) == len(y_test), "Mismatch between X_test and y_test lengths"
                st.write(f"Length of X_test: {len(X_test)}")
                st.write(f"Length of y_test: {len(y_test)}")
                
                # Convert to DataFrame/Series and reset indexes to ensure proper alignment
                X_test = pd.DataFrame(X_test, columns=df.columns[:-1]).reset_index(drop=True)
                y_test = pd.Series(y_test).reset_index(drop=True)
                y_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)
                
                # Get probabilities if supported
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    proba_df = pd.DataFrame(y_proba, columns=[f"Prob_{cls}" for cls in model.classes_])
                else:
                    proba_df = pd.DataFrame()

                # Create a DataFrame to include all relevant information
                predictions_df = X_test.copy()  # This will include the "Length" and "Width" columns from X_test
                predictions_df['Actual'] = y_test
                predictions_df['Predicted'] = y_pred
                
                # Merge probabilities with the main DataFrame if available
                if not proba_df.empty:
                    predictions_df = pd.concat([predictions_df, proba_df], axis=1)
                
                # Display the first few rows to verify correctness
                st.write(predictions_df.head())
                
                # Convert the DataFrame to CSV
                csv = predictions_df.to_csv(index=False)
                
                # Download the CSV file
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )

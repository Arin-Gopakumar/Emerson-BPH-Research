import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats

# Step 1: Enhanced Data Loading and Processing
def load_and_process_data(file_name):
    raw_data = pd.read_csv(file_name, header=None)
    patients = []
    current_patient = {}
    
    for _, row in raw_data.iterrows():
        key = row[0]
        pre = row[1]
        post = row[2] if len(row) > 2 else None
        
        if pd.isna(key) and pd.isna(pre):
            if current_patient:
                patients.append(current_patient)
                current_patient = {}
        elif isinstance(key, str) and "Patient No." in key:
            if current_patient:
                patients.append(current_patient)
            current_patient = {"Patient_No": pre}
        elif key is not None:
            # Extract the base metric name
            if isinstance(key, str):
                # Remove everything up to the colon and clean up the remaining text
                base_metric = key.split(':')[0].strip() if ':' in key else key.strip()
                base_metric = (base_metric.replace("Pre-", "")
                                       .replace("Post-", "")
                                       .replace(" ", "_")
                                       .replace("(", "")
                                       .replace(")", "")
                                       .replace("/", "_per_"))
                
                # Handle special case for maximum flow rate
                if "Maximum_flow_rate" in base_metric:
                    base_metric = "Qmax"
                elif "Maximum_detrusor_pressure" in base_metric:
                    base_metric = "Pdet_max"
                elif "Detrusor_pressure_at_maximum_flow_rate" in base_metric:
                    base_metric = "PdetQmax"
                elif "Bladder_capacity" in base_metric:
                    base_metric = "VH2O_cap"
                
                # Handle multiple measurements in VH2O-cap
                if isinstance(pre, str) and "#1" in pre:
                    pre = float(pre.split("#1 = ")[1].split("\\")[0])
                if isinstance(post, str) and "#1" in post:
                    post = float(post.split("#1 = ")[1].split("\\")[0])
                
                # Store values with consistent naming
                if pre is not None and str(pre).strip():
                    current_patient[f"Pre_{base_metric}"] = pre
                if post is not None and str(post).strip():
                    current_patient[f"Post_{base_metric}"] = post
    
    if current_patient:
        patients.append(current_patient)
    
    df = pd.DataFrame(patients)
    
    # Convert numeric columns
    numeric_columns = [col for col in df.columns if any(metric in col for metric in ['Qmax', 'Pdet_max', 'PdetQmax', 'VH2O_cap'])]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Step 2: Enhanced Data Analysis
def analyze_uds_metrics(df):
    metrics = ['Qmax', 'Pdet_max', 'PdetQmax', 'VH2O_cap']
    analysis_results = {}
    
    for metric in metrics:
        pre_col = f'Pre_{metric}'
        post_col = f'Post_{metric}'
        
        if pre_col in df.columns and post_col in df.columns:
            # Calculate differences and percentage changes
            df[f'{metric}_diff'] = df[post_col] - df[pre_col]
            df[f'{metric}_pct_change'] = ((df[post_col] - df[pre_col]) / df[pre_col] * 100)
            
            # Perform statistical analysis
            valid_pairs = df[[pre_col, post_col]].dropna()
            if len(valid_pairs) >= 2:  # Need at least 2 pairs for t-test
                t_stat, p_value = stats.ttest_rel(valid_pairs[pre_col], 
                                                valid_pairs[post_col])
            else:
                t_stat, p_value = np.nan, np.nan
            
            analysis_results[metric] = {
                'mean_pre': df[pre_col].mean(),
                'std_pre': df[pre_col].std(),
                'mean_post': df[post_col].mean(),
                'std_post': df[post_col].std(),
                'mean_diff': df[f'{metric}_diff'].mean(),
                'mean_pct_change': df[f'{metric}_pct_change'].mean(),
                'p_value': p_value,
                'n_samples': len(valid_pairs)
            }
    
    return df, analysis_results

# Step 3: Enhanced Visualization
def create_visualizations(df, analysis_results):
    metrics = ['Qmax', 'Pdet_max', 'PdetQmax', 'VH2O_cap']
    
    # Print column names for debugging
    print("\nAvailable columns in DataFrame:", df.columns.tolist())
    
    # Create boxplots for pre/post comparisons
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        pre_col = f'Pre_{metric}'
        post_col = f'Post_{metric}'
        
        if pre_col in df.columns and post_col in df.columns:
            plt.subplot(2, 2, i)
            data_to_plot = [
                df[pre_col].dropna(),
                df[post_col].dropna()
            ]
            plt.boxplot(data_to_plot, labels=['Pre', 'Post'])
            plt.title(f'{metric} Comparison\n' +
                     f'n={analysis_results[metric]["n_samples"]}\n' +
                     f'p={analysis_results[metric]["p_value"]:.3f}')
            plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    
    # Create correlation heatmap for pre-treatment metrics
    pre_metrics = [f'Pre_{metric}' for metric in metrics if f'Pre_{metric}' in df.columns]
    if pre_metrics:
        correlation_matrix = df[pre_metrics].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between Pre-Treatment Metrics')
        plt.show()

# Step 4: Enhanced Machine Learning Model
def build_predictive_model(df):
    # Prepare features and target
    metrics = ['Qmax', 'Pdet_max', 'PdetQmax', 'VH2O_cap']
    features = [f'Pre_{metric}' for metric in metrics if f'Pre_{metric}' in df.columns]
    
    if not features:
        print("Error: No valid feature columns found")
        return None, None, None, None, None
    
    # Create multiple target variables for different success criteria
    if 'Qmax_pct_change' in df.columns and 'VH2O_cap_pct_change' in df.columns:
        df['Qmax_significant_improvement'] = (df['Qmax_pct_change'] > 20).astype(int)
        df['Overall_improvement'] = ((df['Qmax_pct_change'] > 20) & 
                                   (df['VH2O_cap_pct_change'] > 10)).astype(int)
    else:
        print("Error: Required percentage change columns not found")
        return None, None, None, None, None
    
    # Prepare the data
    X = df[features].copy()
    y = df['Overall_improvement']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Check if we have enough data
    if len(X) < 10:  # Arbitrary minimum size
        print("Warning: Very small dataset size may affect model reliability")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # Adjust parameter grid based on dataset size
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 3],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    
    # Train model using GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=min(5, len(X)), scoring='f1')
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, importance, X_test, y_test, y_pred

# Main execution
if __name__ == "__main__":
    # Load and process data
    print("Loading and processing data...")
    df = load_and_process_data('Transcribed_Data.csv')
    
    # Print initial data info for debugging
    print("\nDataset Info:")
    print(df.info())
    
    # Analyze metrics
    print("\nAnalyzing metrics...")
    df, analysis_results = analyze_uds_metrics(df)
    
    # Print analysis results
    print("\nAnalysis Results:")
    for metric, results in analysis_results.items():
        print(f"\n{metric}:")
        print(f"Number of valid pairs: {results['n_samples']}")
        print(f"Pre-treatment:  Mean = {results['mean_pre']:.2f} ± {results['std_pre']:.2f}")
        print(f"Post-treatment: Mean = {results['mean_post']:.2f} ± {results['std_post']:.2f}")
        print(f"Mean change: {results['mean_diff']:.2f} ({results['mean_pct_change']:.1f}%)")
        print(f"P-value: {results['p_value']:.3f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, analysis_results)
    
    # Build and evaluate predictive model
    print("\nBuilding predictive model...")
    model_results = build_predictive_model(df)
    
    if all(result is not None for result in model_results):
        model, importance, X_test, y_test, y_pred = model_results
        
        # Print model results
        print("\nModel Evaluation:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nFeature Importance:")
        print(importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance.plot(x='feature', y='importance', kind='bar')
        plt.title('Feature Importance in Predicting Treatment Success')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("\nModel building failed due to data issues")

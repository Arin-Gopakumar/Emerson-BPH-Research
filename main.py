import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy import stats

def load_and_process_data(file_name):
    # Previous load_and_process_data function remains the same
    # [Previous implementation remains unchanged]
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
            if isinstance(key, str):
                base_metric = key.split(':')[0].strip() if ':' in key else key.strip()
                base_metric = (base_metric.replace("Pre-", "")
                                       .replace("Post-", "")
                                       .replace(" ", "_")
                                       .replace("(", "")
                                       .replace(")", "")
                                       .replace("/", "_per_"))
                
                if "Maximum_flow_rate" in base_metric:
                    base_metric = "Qmax"
                elif "Maximum_detrusor_pressure" in base_metric:
                    base_metric = "Pdet_max"
                elif "Detrusor_pressure_at_maximum_flow_rate" in base_metric:
                    base_metric = "PdetQmax"
                elif "Bladder_capacity" in base_metric:
                    base_metric = "VH2O_cap"
                elif "Detrusor_Contractility" in base_metric:
                    base_metric = "Contractility"
                elif "Infusion_Rate" in base_metric:
                    base_metric = "Infusion_Rate"
                
                if isinstance(pre, str) and "#1" in pre:
                    pre = float(pre.split("#1 = ")[1].split("\\")[0])
                if isinstance(post, str) and "#1" in post:
                    post = float(post.split("#1 = ")[1].split("\\")[0])
                
                if pre is not None and str(pre).strip():
                    current_patient[f"Pre_{base_metric}"] = pre
                if post is not None and str(post).strip():
                    current_patient[f"Post_{base_metric}"] = post
    
    if current_patient:
        patients.append(current_patient)
    
    df = pd.DataFrame(patients)
    
    numeric_columns = [col for col in df.columns if any(metric in col for metric in ['Qmax', 'Pdet_max', 'PdetQmax', 'VH2O_cap'])]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def analyze_patient_outcomes(df):
    """Generate patient-specific analysis and clinical interpretations."""
    patient_analyses = []
    
    for _, patient in df.iterrows():
        analysis = {
            'Patient_No': patient['Patient_No'],
            'findings': []
        }
        
        # Analyze Qmax changes
        if 'Pre_Qmax' in patient and 'Post_Qmax' in patient:
            qmax_change = patient['Post_Qmax'] - patient['Pre_Qmax']
            qmax_pct = (qmax_change / patient['Pre_Qmax'] * 100) if patient['Pre_Qmax'] != 0 else np.nan
            
            if qmax_change > 0:
                if qmax_change > 5:
                    analysis['findings'].append(f"Significant improvement in flow rate (↑{qmax_change:.1f} mL/s, {qmax_pct:.1f}%)")
                else:
                    analysis['findings'].append(f"Modest improvement in flow rate (↑{qmax_change:.1f} mL/s, {qmax_pct:.1f}%)")
            else:
                analysis['findings'].append(f"Decreased flow rate (↓{abs(qmax_change):.1f} mL/s, {qmax_pct:.1f}%)")
        
        # Analyze bladder capacity changes
        if 'Pre_VH2O_cap' in patient and 'Post_VH2O_cap' in patient:
            capacity_change = patient['Post_VH2O_cap'] - patient['Pre_VH2O_cap']
            capacity_pct = (capacity_change / patient['Pre_VH2O_cap'] * 100) if patient['Pre_VH2O_cap'] != 0 else np.nan
            
            if abs(capacity_change) > 50:  # Significant change threshold
                direction = "increased" if capacity_change > 0 else "decreased"
                analysis['findings'].append(f"Bladder capacity {direction} by {abs(capacity_change):.1f} mL ({capacity_pct:.1f}%)")
        
        # Analyze detrusor pressure changes
        if 'Pre_Pdet_max' in patient and 'Post_Pdet_max' in patient:
            pdet_change = patient['Post_Pdet_max'] - patient['Pre_Pdet_max']
            if abs(pdet_change) > 20:  # Significant change threshold
                direction = "increased" if pdet_change > 0 else "decreased"
                analysis['findings'].append(f"Maximum detrusor pressure {direction} by {abs(pdet_change):.1f} cmH2O")
        
        # Overall assessment
        if 'Pre_Qmax' in patient and 'Post_Qmax' in patient:
            if patient['Post_Qmax'] > 15:
                analysis['outcome'] = "Good outcome - Normal flow achieved"
            elif patient['Post_Qmax'] > patient['Pre_Qmax'] * 1.5:
                analysis['outcome'] = "Moderate improvement - Significant increase in flow"
            else:
                analysis['outcome'] = "Limited improvement - Consider follow-up evaluation"
        
        patient_analyses.append(analysis)
    
    return patient_analyses

def create_enhanced_visualizations(df, analysis_results):
    """Create improved visualizations with better formatting and clarity."""
    metrics = ['Qmax', 'Pdet_max', 'PdetQmax', 'VH2O_cap']
    metric_labels = {
        'Qmax': 'Maximum Flow Rate (mL/s)',
        'Pdet_max': 'Maximum Detrusor Pressure (cmH2O)',
        'PdetQmax': 'Detrusor Pressure at Qmax (cmH2O)',
        'VH2O_cap': 'Bladder Capacity (mL)'
    }
    
    # Set style for better visualization
    plt.style.use('seaborn')
    
    # 1. Individual Patient Trajectories
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        pre_col = f'Pre_{metric}'
        post_col = f'Post_{metric}'
        
        if pre_col in df.columns and post_col in df.columns:
            ax = axes[idx]
            
            # Plot individual patient trajectories
            for _, patient in df.iterrows():
                ax.plot([1, 2], [patient[pre_col], patient[post_col]], 
                       'gray', alpha=0.3, linewidth=1)
            
            # Plot mean values
            mean_pre = df[pre_col].mean()
            mean_post = df[post_col].mean()
            ax.plot([1, 2], [mean_pre, mean_post], 'b-', linewidth=2, 
                   label='Mean')
            
            # Add error bars for standard deviation
            ax.errorbar([1, 2], 
                       [mean_pre, mean_post],
                       yerr=[df[pre_col].std(), df[post_col].std()],
                       fmt='none', color='b', capsize=5)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Pre-treatment', 'Post-treatment'])
            ax.set_title(f'{metric_labels[metric]}\np={analysis_results[metric]["p_value"]:.3f}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Correlation Matrix with Enhanced Formatting
    pre_metrics = [f'Pre_{metric}' for metric in metrics if f'Pre_{metric}' in df.columns]
    if pre_metrics:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[pre_metrics].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, 
                   vmax=1,
                   fmt='.2f',
                   square=True,
                   cbar_kws={"shrink": .5})
        
        plt.title('Correlation between Pre-Treatment Metrics', pad=20)
        plt.tight_layout()
        plt.show()

def generate_clinical_summary(df, analysis_results, patient_analyses):
    """Generate a comprehensive clinical summary of the findings."""
    summary = []
    
    # Overall cohort characteristics
    summary.append("COHORT OVERVIEW:")
    summary.append(f"Total patients analyzed: {len(df)}")
    
    # Treatment effectiveness
    qmax_improved = df[df['Post_Qmax'] > df['Pre_Qmax']].shape[0]
    summary.append(f"\nTREATMENT EFFECTIVENESS:")
    summary.append(f"- {qmax_improved} out of {len(df)} patients ({qmax_improved/len(df)*100:.1f}%) showed improvement in maximum flow rate")
    
    # Key findings
    summary.append("\nKEY FINDINGS:")
    for metric, results in analysis_results.items():
        if results['mean_diff'] != 0:
            direction = "increase" if results['mean_diff'] > 0 else "decrease"
            sig_level = "Significant" if results['p_value'] < 0.05 else "Non-significant"
            summary.append(f"- {sig_level} {direction} in {metric}: {abs(results['mean_diff']):.1f} units (p={results['p_value']:.3f})")
    
    # Patient-specific notable outcomes
    summary.append("\nNOTABLE PATIENT OUTCOMES:")
    for analysis in patient_analyses:
        if len(analysis['findings']) > 0:
            summary.append(f"\nPatient {analysis['Patient_No']}:")
            for finding in analysis['findings']:
                summary.append(f"- {finding}")
            summary.append(f"Overall: {analysis['outcome']}")
    
    return "\n".join(summary)

# Main execution
if __name__ == "__main__":
    # Load and process data
    print("Loading and processing data...")
    df = load_and_process_data('Transcribed_Data.csv')
    
    # Analyze metrics (using previous analyze_uds_metrics function)
    print("\nAnalyzing metrics...")
    df, analysis_results = analyze_uds_metrics(df)  # Using the previous function
    
    # Generate patient-specific analyses
    patient_analyses = analyze_patient_outcomes(df)
    
    # Create enhanced visualizations
    print("\nGenerating visualizations...")
    create_enhanced_visualizations(df, analysis_results)
    
    # Generate and print clinical summary
    print("\nCLINICAL SUMMARY")
    print("=" * 50)
    clinical_summary = generate_clinical_summary(df, analysis_results, patient_analyses)
    print(clinical_summary)
    
    # Machine learning model (previous implementation remains if needed)
    print("\nBuilding predictive model...")
    model_results = build_predictive_model(df)  # Using the previous function
    
    if all(result is not None for result in model_results):
        model, importance, X_test, y_test, y_pred = model_results
        
        print("\nPREDICTIVE MODEL RESULTS")
        print("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nFeature Importance:")
        print(importance)
        
        # Plot feature importance with improved formatting
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='feature', y='importance')
        plt.title('Feature Importance in Predicting Treatment Success')
        plt.xlabel('Pre-treatment Metrics')
        plt.ylabel('Relative Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
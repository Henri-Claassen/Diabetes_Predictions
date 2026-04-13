# Imports:

import pandas as pd, numpy as np, pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

#Path setup (Sets up paths to data and artifacts directories relative to the project root, which is determined dynamically. This allows for flexible file management and ensures that the code can be run from any location without hardcoding absolute paths.)

Project_Root = Path(__file__).resolve().parents[1]
Data_Dir = Project_Root / 'data'
Artifacts_Dir = Project_Root / 'artifacts'

# Defining columns that need to be changed (Need to be dropped or encoded)

OUTPUT = 'diabetes_stage'

LEAKAGE_COLS = ['diagnosed_diabetes', 'diabetes_risk_score']

ORDINAL_COLS = {
    'education_level': ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
    'income_level':    ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
}

NOMINAL_COLS = [
    'gender',
    'ethnicity',
    'employment_status',
    'smoking_status',
]

BINARY_COLS = [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
]

NUMERIC_COLS = [
    'Age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'glucose_fasting',
    'glucose_postprandial',
    'insulin_level',
    'hba1c',
]

FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + list(ORDINAL_COLS.keys()) + NOMINAL_COLS

# Load already split data
def load_data():
    train = pd.read_csv(Data_Dir / 'train.csv')
    test = pd.read_csv(Data_Dir / 'test.csv')
    return train, test

# Drop leakage columns
# We drop columns that wouldnt be available in real world scenarios to prevent data leakage and ensure our models learn to predict diabetes stage based on features that would actually be available at the time of prediction. We check if the columns exist before dropping to avoid errors if they are not present in the dataset. We return the cleaned dataframes.
def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

#Encode the Output column
#Changing the string labels to numeric values that the models can understand. We use LabelEncoder for the output column because it is a classification problem and the output is nominal. We save the encoder so we can use it later in the dashboard to decode model predictions back to the original string labels. We return the encoded output arrays and the encoder.
def encode_output(train: pd.DataFrame, test: pd.DataFrame):
    le = LabelEncoder()
    y_train = le.fit_transform(train[OUTPUT]) #This learns the mapping from train
    y_test = le.transform(test[OUTPUT]) #This applies the same mapping to test

    with open(Artifacts_Dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print(f"Output class: {dict(enumerate(le.classes_))}")
    return y_train, y_test, le

#Encode the categorical features
#This means we change the string values to numeric values that the models can understand. We use OrdinalEncoder for the ordinal columns, which have a natural order, and LabelEncoder for the nominal columns, which do not have a natural order. We save the encoders so we can use them later in the dashboard to encode user input in the same way as the training data. We return the encoded feature matrices and the encoders.
def encode_features(train: pd.DataFrame, test: pd.DataFrame):
    #We make copies to not edit raw dataframes

    X_train = train[FEATURE_COLS].copy() #Making dataframe of train inputs
    X_test = test[FEATURE_COLS].copy() #Making dataframe of test inputs

    encoders = {} #Save this to the dashboard later for consistent encoding of user input

    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_COLS[col] for col in ORDINAL_COLS],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    #Ordinal Columns
    ordinal_col_list = list(ORDINAL_COLS.keys())
    X_train[ordinal_col_list] = ordinal_encoder.fit_transform(X_train[ordinal_col_list])
    X_test[ordinal_col_list] = ordinal_encoder.transform(X_test[ordinal_col_list])
    encoders['ordinal'] = ordinal_encoder

    #Nominal Columns
    nominal_encoders = {}

    for col in NOMINAL_COLS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        nominal_encoders[col] = le

    encoders['nominal'] = nominal_encoders

    with open(Artifacts_Dir / 'feature_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Encoded matrix shapes train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, encoders

#Scaling
#This function scales the features using StandardScaler and saves the scaler for later use in the dashboard and inference. We scale the features for K-Means because it is sensitive to feature scales, while tree-based models are not. We return both scaled and unscaled versions of the data so we can use the appropriate one for each model type.
def scale_for_kmeans(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Convert back to dataframe so column names are preserved
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    with open(Artifacts_Dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaled matrix shapes train: {X_train_scaled.shape}, test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler    
    
# Save Processed Data
def save_processed_data(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    X_train.to_csv(Data_Dir / 'X_train.csv', index=False)
    X_test.to_csv(Data_Dir / 'X_test.csv', index=False)

    pd.Series(y_train, name=OUTPUT).to_csv(Data_Dir / 'y_train.csv', index=False)
    pd.Series(y_test, name=OUTPUT).to_csv(Data_Dir / 'y_test.csv', index=False)

    X_train_scaled.to_csv(Data_Dir / 'X_train_scaled.csv', index=False)
    X_test_scaled.to_csv(Data_Dir / 'X_test_scaled.csv', index=False)
    print("Saved processed data to data directory.")

# Running all of the functions to create the processed data and save it
def preprocess_data():
    train, test = load_data()

    train = drop_leakage(train)
    test = drop_leakage(test)

    y_train, y_test, output_encoder = encode_output(train, test)

    X_train, X_test, feature_encoders = encode_features(train, test)

    X_train_scaled, X_test_scaled, scaler = scale_for_kmeans(X_train, X_test)

    save_processed_data(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    return{
        #Unscaled used for (Decision tree, random forest, xgboost)
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,

        #Scaled used for K-Means
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        #Encoders used for dashboard / inference

        'output_encoder': output_encoder,
        'feature_encoders': feature_encoders,
        'scaler': scaler
    }

if __name__ == '__main__':
    artifacts = preprocess_data()
   

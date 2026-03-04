import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    df = pd.read_csv(r'c:\Users\HP\Desktop\Malik\malik2\Loan_Default (1).csv')
    categorical_cols = ['Gender', 'loan_type', 'loan_purpose', 'occupancy_type', 'credit_type', 'age', 'Region']
    
    mapping_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Dropna so we don't encode floats and NaNs weirdly, or convert to str
        encoded = le.fit_transform(df[col].astype(str))
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        mapping_dict[col] = mapping
        
    for col, map_val in mapping_dict.items():
        print(f"'{col}': {map_val},")
        
except Exception as e:
    print(f"Error: {e}")

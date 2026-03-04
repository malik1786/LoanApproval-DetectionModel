import joblib
import pandas as pd
import numpy as np

FEATURES = [
    'ID', 'year', 'Gender', 'loan_type', 'loan_purpose', 'loan_amount', 'rate_of_interest', 
    'Interest_rate_spread', 'Upfront_charges', 'term', 'property_value', 'occupancy_type', 
    'total_units', 'income', 'credit_type', 'Credit_Score', 'age', 'LTV', 'Region', 'dtir1', 
    'loan_limit_ncf', 'approv_in_adv_pre', 'Credit_Worthiness_l2', 'open_credit_opc', 
    'business_or_commercial_nob/c', 'Neg_ammortization_not_neg', 'interest_only_not_int', 
    'lump_sum_payment_not_lpsm', 'construction_type_sb', 'Secured_by_land', 
    'co-applicant_credit_type_EXP', 'submission_of_application_to_inst', 'Security_Type_direct'
]

# "Risky" profile
data1 = {f: 0 for f in FEATURES}
data1['Credit_Score'] = 30
data1['income'] = 500
data1['loan_amount'] = 10000000
data1['rate_of_interest'] = 50
data1['term'] = 36

# "Safe" profile
data2 = {f: 0 for f in FEATURES}
data2['Credit_Score'] = 800
data2['income'] = 10000
data2['loan_amount'] = 50000
data2['rate_of_interest'] = 5
data2['term'] = 360

df1 = pd.DataFrame([data1])[FEATURES]
df2 = pd.DataFrame([data2])[FEATURES]

try:
    model = joblib.load('final_model.pkl')
    print("Classes:", model.classes_)
    
    pred1 = model.predict(df1)[0]
    prob1 = model.predict_proba(df1)[0] if hasattr(model, 'predict_proba') else None
    
    pred2 = model.predict(df2)[0]
    prob2 = model.predict_proba(df2)[0] if hasattr(model, 'predict_proba') else None
    
    print(f"Risky Profile -> Pred: {pred1}, Probs: {prob1}")
    print(f"Safe Profile  -> Pred: {pred2}, Probs: {prob2}")

except Exception as e:
    print(f"Error: {e}")

import pandas as pd

try:
    df = pd.read_csv(r'c:\Users\HP\Desktop\Malik\malik2\Loan_Default (1).csv')
    cols_to_check = [
        'loan_limit', 'approv_in_adv', 'Credit_Worthiness', 'open_credit', 
        'business_or_commercial', 'Neg_ammortization', 'interest_only', 
        'lump_sum_payment', 'construction_type', 'Secured_by', 
        'co-applicant_credit_type', 'submission_of_application', 'Security_Type'
    ]
    
    for col in cols_to_check:
        print(f"{col}: {df[col].unique()}")
        
except Exception as e:
    print(f"Error: {e}")

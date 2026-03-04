import pandas as pd

try:
    df = pd.read_csv(r'c:\Users\HP\Desktop\Malik\malik2\Loan_Default (1).csv')
    print("Status counts:")
    print(df['Status'].value_counts())
    
    print("\nStatus vs Credit Score Mean:")
    print(df.groupby('Status')['Credit_Score'].mean())

    print("\nStatus vs Income Mean:")
    print(df.groupby('Status')['income'].mean())
    
except Exception as e:
    print(f"Error: {e}")

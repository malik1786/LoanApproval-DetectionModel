from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('final_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature list based on inspection
FEATURES = [
    'ID', 'year', 'Gender', 'loan_type', 'loan_purpose', 'loan_amount', 'rate_of_interest', 
    'Interest_rate_spread', 'Upfront_charges', 'term', 'property_value', 'occupancy_type', 
    'total_units', 'income', 'credit_type', 'Credit_Score', 'age', 'LTV', 'Region', 'dtir1', 
    'loan_limit_ncf', 'approv_in_adv_pre', 'Credit_Worthiness_l2', 'open_credit_opc', 
    'business_or_commercial_nob/c', 'Neg_ammortization_not_neg', 'interest_only_not_int', 
    'lump_sum_payment_not_lpsm', 'construction_type_sb', 'Secured_by_land', 
    'co-applicant_credit_type_EXP', 'submission_of_application_to_inst', 'Security_Type_direct'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Prepare the input dataframe mapping raw inputs to model's 33 exact features
        input_data = {}
        
        # Statistical Modes from Dataset (Baselines)
        MODES = {
            'year': 2019.0, 'loan_limit': 'cf', 'Gender': 'Male', 'approv_in_adv': 'nopre', 
            'loan_type': 'type1', 'loan_purpose': 'p3', 'Credit_Worthiness': 'l1', 
            'open_credit': 'nopc', 'business_or_commercial': 'nob/c', 'loan_amount': 206500.0, 
            'rate_of_interest': 3.99, 'term': 360.0, 'Neg_ammortization': 'not_neg', 
            'interest_only': 'not_int', 'lump_sum_payment': 'not_lpsm', 'property_value': 308000.0, 
            'construction_type': 'sb', 'occupancy_type': 'pr', 'Secured_by': 'home', 
            'total_units': '1U', 'income': 5000.0, 'credit_type': 'CIB', 'Credit_Score': 763.0, 
            'co-applicant_credit_type': 'CIB', 'age': '45-54', 'submission_of_application': 'to_inst', 
            'LTV': 81.25, 'Region': 'North', 'Security_Type': 'direct', 'dtir1': 37.0
        }

        # Helper to parse currency shorthand (1k, 1lac, 1cr etc)
        def parse_currency(val):
            if isinstance(val, (int, float)): return val
            if not val or not isinstance(val, str): return 0
            
            val = val.lower().replace(',', '').strip()
            # Remove currency symbols if present
            val = val.replace('$', '').replace('rs', '').replace('₹', '')
            
            multipliers = {
                'k': 1000,
                'l': 100000,
                'lac': 100000,
                'lakh': 100000,
                'cr': 10000000,
                'crore': 10000000,
                'm': 1000000,
                'b': 1000000000
            }
            
            # Check for matches
            for unit, mult in multipliers.items():
                if val.endswith(unit):
                    try:
                        return float(val.replace(unit, '')) * mult
                    except:
                        continue
            
            try:
                return float(val)
            except:
                return 0

        # Base numerical conversion helper
        def get_val(key, data, defaults):
            val = data.get(key)
            if val is None or val == "":
                return defaults.get(key, 0)
            
            # Smart parsing for financial numeric fields
            if key in ['loan_amount', 'income', 'property_value']:
                return parse_currency(val)
                
            return val

        # Mapping dictionaries for LabelEncoders
        label_encoders = {
            'Gender': {'Female': 0, 'Joint': 1, 'Male': 2, 'Sex Not Available': 3},
            'loan_type': {'type1': 0, 'type2': 1, 'type3': 2},
            'loan_purpose': {'p1': 0, 'p2': 1, 'p3': 2, 'p4': 3},
            'occupancy_type': {'ir': 0, 'pr': 1, 'sr': 2},
            'credit_type': {'CIB': 0, 'CRIF': 1, 'EQUI': 2, 'EXP': 3},
            'age': {'25-34': 0, '35-44': 1, '45-54': 2, '55-64': 3, '65-74': 4, '<25': 5, '>74': 6},
            'Region': {'North': 0, 'North-East': 1, 'central': 2, 'south': 3}
        }

        for feat in FEATURES:
            raw_val = get_val(feat, data, MODES)
            
            # 1. Numerical Mappings
            if feat in ['year', 'loan_amount', 'rate_of_interest', 'Interest_rate_spread', 
                        'Upfront_charges', 'term', 'property_value', 'income', 
                        'Credit_Score', 'LTV', 'dtir1', 'ID']:
                try:
                    input_data[feat] = [float(raw_val)]
                except:
                    input_data[feat] = [float(MODES.get(feat, 0))]
            
            # 2. Sequential Categorical
            elif feat == 'total_units':
                units_map = {'1U': 0, '2U': 1, '3U': 2, '4U': 3}
                input_data[feat] = [units_map.get(raw_val, 0)]

            elif feat in label_encoders:
                input_data[feat] = [label_encoders[feat].get(raw_val, 0)]

            # 3. Binary Indicator Mappings (Model-specific Suffixes)
            elif feat == 'loan_limit_ncf':
                input_data[feat] = [1 if get_val('loan_limit', data, MODES) == 'ncf' else 0]
            elif feat == 'approv_in_adv_pre':
                input_data[feat] = [1 if get_val('approv_in_adv', data, MODES) == 'pre' else 0]
            elif feat == 'Credit_Worthiness_l2':
                input_data[feat] = [1 if get_val('Credit_Worthiness', data, MODES) == 'l2' else 0]
            elif feat == 'open_credit_opc':
                input_data[feat] = [1 if get_val('open_credit', data, MODES) == 'opc' else 0]
            elif feat == 'business_or_commercial_nob/c':
                input_data[feat] = [1 if get_val('business_or_commercial', data, MODES) == 'nob/c' else 0]
            elif feat == 'Neg_ammortization_not_neg':
                input_data[feat] = [1 if get_val('Neg_ammortization', data, MODES) == 'not_neg' else 0]
            elif feat == 'interest_only_not_int':
                input_data[feat] = [1 if get_val('interest_only', data, MODES) == 'not_int' else 0]
            elif feat == 'lump_sum_payment_not_lpsm':
                input_data[feat] = [1 if get_val('lump_sum_payment', data, MODES) == 'not_lpsm' else 0]
            elif feat == 'construction_type_sb':
                input_data[feat] = [1 if get_val('construction_type', data, MODES) == 'sb' else 0]
            elif feat == 'Secured_by_land':
                input_data[feat] = [1 if get_val('Secured_by', data, MODES) == 'land' else 0]
            elif feat == 'co-applicant_credit_type_EXP':
                input_data[feat] = [1 if get_val('co-applicant_credit_type', data, MODES) == 'EXP' else 0]
            elif feat == 'submission_of_application_to_inst':
                input_data[feat] = [1 if get_val('submission_of_application', data, MODES) == 'to_inst' else 0]
            elif feat == 'Security_Type_direct':
                input_data[feat] = [1 if get_val('Security_Type', data, MODES) == 'direct' else 0]
            
            else:
                input_data[feat] = [0]

        df = pd.DataFrame(input_data)
        df = df[FEATURES]
        
        # Prediction and Probability
        prediction = model.predict(df)[0]
        prob = 0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            prob = float(probs[1]) if len(probs) > 1 else (float(probs[0]) if prediction == 1 else 0)

        # Expert Banking Safety Rails (Manual Overrides for extreme cases)
        # If Credit Score is extremely low or Debt-to-Income is extremely high
        credit_score = df['Credit_Score'].iloc[0]
        income = df['income'].iloc[0]
        loan_amount = df['loan_amount'].iloc[0]
        
        is_expert_red_flag = (credit_score < 400) or (income < 1000 and loan_amount > 50000)
        
        # UI Response Logic
        is_risky = (prediction == 1) or (prob > 0.4) or is_expert_red_flag
        
        # --- Reasoning Engine ---
        reasons = []
        ltv = df['LTV'].iloc[0]
        dtir = df['dtir1'].iloc[0]

        if is_risky:
            if credit_score < 450:
                reasons.append("Critically low credit score (below 450) indicates high historical default risk.")
            elif credit_score < 600:
                reasons.append("Sub-prime credit score (below 600) suggests inconsistent repayment patterns.")
            
            if income < 1500:
                reasons.append("Monthly income is insufficient to safely cover loan obligations and living expenses.")
            
            if ltv > 100:
                reasons.append("Loan-to-Value (LTV) exceeds 100%, meaning the loan is larger than the collateral value.")
            
            if dtir > 50:
                reasons.append("Debt-to-Income ratio (DTI) is dangerously high, exceeding 50% of monthly gross income.")
            
            if is_expert_red_flag and len(reasons) == 0:
                reasons.append("Profile contains extreme high-risk anomalies that deviate from banking safety standards.")
            
            if len(reasons) == 0:
                reasons.append("Mathematical risk modeling identifies patterns statistically correlated with high default rates.")

            # Severity labeling
            if prob > 0.65 or (is_expert_red_flag and credit_score < 350):
                result_label = "🚨 CRITICAL: Very High Risk"
                result_class = "critical-risk"
                desc = "Extreme priority flags detected. Massive deviation from baseline stability."
            else:
                result_label = "High Risk (Needs Review)"
                result_class = "high-risk"
                desc = "This applicant shows significant risk factors. Manual verification is mandatory."
        else:
            # Approval Reasons
            if credit_score > 700:
                reasons.append("Strong credit score demonstrates excellent financial reliability.")
            if income > 5000:
                reasons.append("Healthy monthly income provides a safe margin for repayment.")
            if ltv < 80:
                reasons.append("Low Loan-to-Value (LTV) ratio provides strong collateral backing.")
            if dtir < 35:
                reasons.append("Low Debt-to-Income ratio indicates high manageable borrowing capacity.")
            
            if len(reasons) == 0:
                reasons.append("Applicant profile meets standard safety thresholds for automated approval.")

            result_label = "Low Risk (Approved)"
            result_class = "low-risk"
            desc = "Verification complete. Applicant meets the bank's safety threshold for approval."
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(round(prob * 100, 1)),
            'label': result_label,
            'class': result_class,
            'description': desc,
            'reasons': reasons,
            'red_flag': bool(is_expert_red_flag)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

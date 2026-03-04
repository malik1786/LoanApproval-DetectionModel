import joblib

try:
    model = joblib.load('final_model.pkl')
    if hasattr(model, 'feature_names_in_'):
        print("FEATURES = ", list(model.feature_names_in_))
    else:
        print("Model does not have feature_names_in_")
except Exception as e:
    print(f"Error: {e}")

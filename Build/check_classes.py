import joblib

try:
    model = joblib.load('final_model.pkl')
    print("Classes:", model.classes_)
except Exception as e:
    print(f"Error: {e}")

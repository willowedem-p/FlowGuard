# Smart Water Quality Assessment System

## How to run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the app:
   streamlit run app.py

The app will open in your browser at http://localhost:8501

## Files
- app.py            -> Main Streamlit application (front page, dashboard, history, recommendations, print)
- model.pkl         -> Trained Logistic Regression model (from your notebook code, unchanged)
- scaler.pkl        -> StandardScaler fitted on the training data
- water_quality_dataset1.csv -> Original dataset used for training
- style.css         -> App styling
- print.css         -> Styling used for the printable report popup
- test_history.csv  -> Auto-created once you run your first test; stores all test results

## Notes
- All ML/engineering logic (ml_prediction, calculate_purity, classify_water,
  recommend_treatment, decision_fusion) is copied directly from your notebook
  and was not modified.
- Every test you run is saved with date & time to test_history.csv, viewable
  under "Previous Tests" and "Recommendations".
- Each result has a "Print this result" button that opens a clean, printable
  report in a new tab.

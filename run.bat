@echo off
echo ========================================================
echo        Starting CardioGuard ML Application
echo ========================================================
echo.
echo 1. Checking dependencies...
python -m pip install -r requirements.txt
echo.
echo 2. Training model and generating evaluation metrics...
python train_model.py
echo.
echo 3. Launching Streamlit Dashboard...
streamlit run app.py
pause

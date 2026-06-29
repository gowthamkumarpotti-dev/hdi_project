# HDI Predictor - Flask Web Application

## Team Members
- Potti Gowtham
- V S Akhil Roy
- Patra Abhilash
- Madikiri Sahana
- Sareddy Pranay Kumar Reddy

---

## HOW TO RUN (Step by Step)

### Step 1 - Install Python
Download Python 3.x from https://python.org and install it.

### Step 2 - Open Command Prompt / Terminal
Navigate to this project folder:
```
cd hdi_project
```

### Step 3 - Install Required Libraries
```
pip install -r requirements.txt
```

### Step 4 - Train the Model (Run ONCE)
```
python train_model.py
```
This creates: model.pkl, hdi_dataset.csv, and 3 plot images.

### Step 5 - Run the Flask App
```
python app.py
```

### Step 6 - Open in Browser
Go to: http://127.0.0.1:5000

---

## Pages
- `/`             → Home page
- `/predict`      → Enter values and get HDI prediction
- `/visualizations` → View EDA plots

---

## Model Performance
- Algorithm: Linear Regression
- R² Score: 0.97 (97% accuracy)
- Features: Life Expectancy, Mean Years of Schooling, GNI Per Capita
- Output: HDI Score (0 to 1) + Category (Very High / High / Medium / Low)

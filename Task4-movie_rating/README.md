#  Task 4: Movie Rating Prediction 🎯

This project is part of the CODSOFT Internship - Task 4.  
The goal is to build a machine learning model that can predict **IMDb movie ratings** based on features like genre, director, actors, duration, votes, etc.

---

##  Dataset

- Source: `data/IMDB-Movie-Data.csv`
- Columns used: `Name, Year, Duration, Genre, Rating, Votes, Director, Actor 1, Actor 2, Actor 3`

---

##  Tech Stack

- **Python 3**
- **scikit-learn**
- **pandas**
- **matplotlib**
- **pickle**

---

##  ML Model

- Model Used: `RandomForestRegressor`
- Preprocessing includes:
  - Handling missing values
  - Genre encoding using `CountVectorizer`
  - Top 10 frequent Directors and Actors encoding
  - Feature extraction from year, duration, votes

---

##  Evaluation Metrics

| Metric | Value |
|--------|-------|
| RMSE   | X.XX  |
| R²     | X.XX  |

>  Actual results will print in the console after training.

---

##  Outputs

-  Feature importance graph is saved in:  
  `output/feature_importance.png`

---

##  How to Run

```bash
# Step 1: Install requirements (if needed)
pip install -r requirements.txt

# Step 2: Run model training
python src/train_model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('static/plots', exist_ok=True)

# ── Synthetic HDI dataset (190 countries approximated) ──────────────────────
np.random.seed(42)
n = 200

life_exp   = np.random.uniform(45, 85, n)
schooling  = np.random.uniform(4, 20, n)
gni        = np.random.uniform(1000, 80000, n)

# HDI formula approximation
hdi_score = (
    0.4 * (life_exp - 20) / (85 - 20) +
    0.3 * (schooling / 20) +
    0.3 * (np.log(gni) - np.log(100)) / (np.log(75000) - np.log(100))
)
hdi_score = np.clip(hdi_score, 0, 1)

def classify(h):
    if h >= 0.8:  return 'Very High'
    elif h >= 0.7: return 'High'
    elif h >= 0.55: return 'Medium'
    else:           return 'Low'

hdi_category = [classify(h) for h in hdi_score]

df = pd.DataFrame({
    'Life_Expectancy': life_exp,
    'Mean_Years_Schooling': schooling,
    'GNI_Per_Capita': gni,
    'HDI_Score': hdi_score,
    'HDI_Category': hdi_category
})

df.to_csv('hdi_dataset.csv', index=False)
print("Dataset created:", df.shape)
print(df['HDI_Category'].value_counts())

# ── EDA Plots ────────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
sns.heatmap(df[['Life_Expectancy','Mean_Years_Schooling','GNI_Per_Capita','HDI_Score']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/plots/heatmap.png', dpi=80)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Life_Expectancy', y='HDI_Score', hue='HDI_Category', palette='Set2')
plt.title('Life Expectancy vs HDI Score')
plt.tight_layout()
plt.savefig('static/plots/scatter.png', dpi=80)
plt.close()

plt.figure(figsize=(8,5))
df['HDI_Score'].hist(bins=20, color='steelblue', edgecolor='black')
plt.title('HDI Score Distribution')
plt.xlabel('HDI Score')
plt.tight_layout()
plt.savefig('static/plots/distribution.png', dpi=80)
plt.close()

# ── Model Training ───────────────────────────────────────────────────────────
X = df[['Life_Expectancy', 'Mean_Years_Schooling', 'GNI_Per_Capita']]
y = df['HDI_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\nR² Score : {r2:.4f}")
print(f"MSE      : {mse:.6f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as model.pkl")

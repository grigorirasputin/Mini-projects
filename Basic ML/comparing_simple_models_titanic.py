import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- 1. Feature Engineering Function ---
def engineer_features(df):
    """Applies advanced feature engineering to extract human context."""
    df = df.copy()
    
    # Extract Titles from Name
    df['Title'] = df['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1))
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Extract Deck from Cabin
    df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0])
    
    # Calculate Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Drop raw columns
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    
    return df

# --- 2. Load and Prepare Data ---
raw_df = pd.read_csv(r"C:\Users\rumia\OneDrive\Documents\GitHub\Mini-projects\Basic ML\Data\titanic\train.csv")
engineered_df = engineer_features(raw_df)

target = 'Survived'
X = engineered_df.drop(columns=[target])
y = engineered_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Build the Preprocessing Pipeline ---
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 4. Define ALL Models ---
models = {
    'Linear Reg': LinearRegression(),
    'Logistic Reg': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=8),
    'MLP': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(50, 50)),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', max_depth=5),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0, depth=5) 
}

# --- 5. Train and Evaluate ---
results = []

print("Training all models on ADVANCED engineered features...")
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    clf.fit(X_train, y_train)
    
    if name == 'Linear Reg':
        preds_continuous = clf.predict(X_test)
        preds = (preds_continuous >= 0.5).astype(int) 
        probs = preds_continuous 
    else:
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'F1 Score': f1_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'AUPRC': average_precision_score(y_test, probs)
    })

# --- 6. Plotting ---
df_results = pd.DataFrame(results)
df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Using a wider figure to comfortably fit 7 bars per metric
plt.figure(figsize=(18, 8)) 
sns.set_theme(style="whitegrid")

ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="tab10")

plt.title('Algorithm Comparison (All Models) on ADVANCED Features', fontsize=16, pad=20)
plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
plt.xlabel('')
plt.ylim(0, 1.1) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=8)

plt.tight_layout()
plt.savefig(r"Plots/Metrics_across_models.png",dpi = 300 )

plt.show()



# --- 7. Interpret the Linear Regression Model ---

# 1. Re-fit the Linear Regression pipeline specifically so we can extract its parts
lin_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
lin_reg_pipeline.fit(X_train, y_train)

# 2. Extract the trained model and the fitted preprocessor
trained_lin_reg = lin_reg_pipeline.named_steps['model']
fitted_preprocessor = lin_reg_pipeline.named_steps['preprocessor']

# 3. Extract the feature names
# Numeric features remain the same
num_cols = numeric_features 
# Categorical features expanded during One-Hot Encoding, so we must ask the encoder for the new names
cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_cols = cat_encoder.get_feature_names_out(categorical_features)

# Combine them in the exact order the ColumnTransformer outputs them
all_feature_names = np.concatenate([num_cols, cat_cols])

# 4. Extract the coefficients
coefficients = trained_lin_reg.coef_

# 5. Create a DataFrame for easy sorting and plotting
coef_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coefficients
})

# Sort by absolute magnitude to find the most impactful features (positive or negative)
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# 6. Plot the top positive and top negative coefficients
plt.figure(figsize=(12, 10))
sns.set_theme(style="whitegrid")

# Create a color palette: Green for positive impact (survival), Red for negative (death)
colors = ['#2ca02c' if c > 0 else '#d62728' for c in coef_df['Coefficient']]

ax = sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette=colors)

plt.title('Linear Regression Coefficients: What drove survival?', fontsize=16, pad=20)
plt.xlabel('Impact on Prediction (Positive = Survived, Negative = Died)', fontsize=12)
plt.ylabel('Engineered Feature', fontsize=12)

# Add a vertical line at 0 for visual clarity
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(r"Plots/feature_importance_linreg.png",dpi = 300 )
plt.show()
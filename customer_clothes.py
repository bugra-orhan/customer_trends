# Step 0: Import the required libraries and read the data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("shopping_trends.csv")

# Step 1: Take the first look at the dataset.
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df.info())
print(df.describe())

# Step 2: Univariate Feature Analysis
# Step 2.1: Determining numerical and categorical features.
num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols = [col for col in num_cols if col!="Customer ID"]

cat_cols = df.select_dtypes(include="object").columns.tolist()

cat_cols_vis = [col for col in cat_cols if col not in ["Color","Item Purchased","Location"]]
cols_barchart = ["Color","Item Purchased","Location"]
print(num_cols)
print(cat_cols)

# Step 2.2: Categorical Features Analysis
n_cat_col_vis = 3
n_cat_row_vis = -(-len(cat_cols_vis)//n_cat_col_vis)

fig, axes = plt.subplots(n_cat_row_vis,n_cat_col_vis,figsize=(n_cat_row_vis*5,n_cat_col_vis*5))
axes = axes.flatten()
for i,col in enumerate(cat_cols_vis):
    df[col].value_counts().plot.pie(
        ax=axes[i],
        autopct = "%1.1f%%",
        startangle = 90)
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("")

for j in range(len(cat_cols_vis), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# Step 2.3: Numerical Features Analysis
n_num_cols = 2
n_num_rows = -(-len(num_cols)//n_num_cols)

fig, axes = plt.subplots(n_num_rows,n_num_cols,figsize=(n_num_cols*5,n_num_rows*5))
axes = axes.flatten()

for i,col in enumerate(num_cols):
    axes[i].hist(df[col],bins=20,edgecolor="black")
    axes[i].set_title(f"{col} Histogram")
    axes[i].set_ylabel("Count")

for j in range(len(num_cols),len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# This is for categorical features that are better to be represented in barcharts
n_cols = 2
n_rows = -(-len(cols_barchart)//n_cols)

fig,axes = plt.subplots(n_rows,n_cols,figsize=(n_cols*5,n_rows*5))
axes = axes.flatten()

for i,col in enumerate(cols_barchart):
    value_counts = df[col].value_counts().head(10)
    axes[i].bar(value_counts.index,value_counts.values)
    axes[i].set_title(f"Top 10 {col}")
    axes[i].tick_params(axis="x",rotation=45)

for j in range(len(cols_barchart),len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# Step 3: Univariate Analysis
# Step 3.1: Categorical vs Numerical Analysis
# Which gender tends to purchase more?
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Gender", y="Purchase Amount (USD)")
plt.title("Purchase Amount by Gender")
plt.show()
# Step 3.1.2: Which categories are higher rated?
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Category", y="Review Rating")
plt.title("Review Rating by Category")
plt.show()
# Step 3.1.3: Purchasing by Season
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Season", y="Purchase Amount (USD)")
plt.title("Purchase Amount by Season")
plt.show()
# Step 3.1.4: Subscription by Previous Purchases
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Subscription Status", y="Previous Purchases")
plt.title("Previous Purchases by Subscription")
plt.show()

# Step 3.2: Categorical vs Categorical Analysis
# Step 3.2.1: Which gender prefers which payment method?
ct = pd.crosstab(df["Payment Method"],df["Gender"])
ct.plot(kind="bar",stacked=True,figsize=(10,6))
plt.title('Payment Method by Gender')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Step 3.2.2: Do subscribers get more discounts?
sns.countplot(data=df, x="Subscription Status", hue="Discount Applied")
plt.title("Discount Applied by Subscription Status")
plt.show()


# Step 3.2.3: Which category is sold in which season?
ct = pd.crosstab(df["Season"],df["Category"])
ct.plot(kind="bar",stacked=True,figsize=(10,6))
plt.title('Category by Season')
plt.xlabel('Category')
plt.ylabel('Count')
plt.legend(title='Category')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Step 3.3: Numerical vs Numerical Analysis
# Step 3.3.1: Does increase in age reflect in purchases?

sns.scatterplot(data=df, x="Age", y="Purchase Amount (USD)")
plt.title("Age vs Purchase Amount")
plt.show()

# Step 3.3.2: Do loyal customers give higher rates?
sns.scatterplot(data=df, x="Previous Purchases", y="Review Rating")
plt.title("Review Rating vs Previous Purchases")
plt.show()

# Step 5: Machine Learning (Binary classification on subscription)
# Step 5.0: Importing required modules for the classification algorithm.

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.compose import ColumnTransformer

# Step 5.1: Separating target feature and Label Encoding 
y = df["Subscription Status"]
X = df.drop(["Customer ID","Subscription Status","Promo Code Used"],axis=1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 5.2: One-Hot Encoding
cats = X.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(handle_unknown="ignore"),cats)],
        remainder="passthrough"
)
X_encoded = preprocessor.fit_transform(X)

# Step 5.3 Splitting train and test data, establishing Random Forest Classification
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

# Step 5.4: The performance measures.
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# Step 6: Cross-Validation

rf_model = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf_model,X_train,y_train,cv=5, scoring="accuracy")

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))

# Step 7: Feature Importance
feature_names = preprocessor.get_feature_names_out()

importances = rf.feature_importances_

feat_imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(5)

plt.figure(figsize=(10,8))
bars = plt.barh(feat_imp_series.index,feat_imp_series.values)
plt.title("Random Forest - 5 Most Important Feature")
plt.xlabel("Importance")
plt.ylabel("Feature")

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center')

plt.show()
from flask import Flask, render_template
import pandas as pd
import numpy as np  
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
import warnings
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)

def load_and_preprocess_data():
    # Load the data
    df = pd.read_csv('boxoffice.csv', encoding='latin-1')

    # Data preprocessing steps
    to_remove = ['world_revenue', 'opening_revenue']
    df.drop(to_remove, axis=1, inplace=True)

    df.drop('budget', axis=1, inplace=True)

    for col in ['MPAA']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['domestic_revenue'] = df['domestic_revenue'].str[1:]
    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        df[col] = df[col].str.replace(',', '')
        temp = (~df[col].isnull())
        df[temp][col] = df[temp][col].astype(float)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # One-hot encoding for genres
    df = pd.get_dummies(df, columns=['genres'], drop_first=True)

    # Label Encoding for 'distributor' and 'MPAA'
    for col in ['distributor', 'MPAA']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Handling NaN values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)

    # Replacing the old numeric columns with imputed ones
    df[df_imputed.columns] = df_imputed

    return df

def train_models(df):
    # Split data into features and target
    features = df.drop(['title', 'domestic_revenue'], axis=1)
    target = df['domestic_revenue'].values

    # Train-test split
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

    # Normalizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Initialize models
    models = {
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "XGBRegressor": XGBRegressor()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, Y_train)
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        results.append({
            'Model': name,
            'Training MAE': mae(Y_train, train_preds),
            'Validation MAE': mae(Y_val, val_preds)
        })

    return results

def plot_results(results):
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sb.barplot(x='Model', y='Training MAE', data=results_df)
    plt.title('Training MAE Comparison')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sb.barplot(x='Model', y='Validation MAE', data=results_df)
    plt.title('Validation MAE Comparison')
    plt.xticks(rotation=45)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return plot_url

@app.route('/')
def index():
    df = load_and_preprocess_data()
    results = train_models(df)
    plot_url = plot_results(results)

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)

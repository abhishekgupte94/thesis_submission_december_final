from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest

def select_features(X, y, k=0.5):
    # XGBoost feature importance
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X, y)
    xgb_importance = xgb.feature_importances_

    # Random Forest feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    # Average feature importance
    avg_importance = (xgb_importance + rf_importance) / 2

    # Select top k% features
    k_features = int(X.shape[1] * k)
    selector = SelectKBest(lambda X, y: avg_importance, k=k_features)
    X_selected = selector.fit_transform(X, y)

    return X_selected, selector

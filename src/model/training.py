from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



def data_split(df):
    X = df[['dist_to_restaurant', 'Hdist_to_restaurant', 'avg_Hdist_to_restaurants',	'date_day_number', 'restaurant_id', 'Five_Clusters_embedding', 'h3_index','date_hour_number',		'restaurants_per_index']]
    y = df[['orders_busyness_by_h3_hour']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


def model_init_and_fit(X_train, y_train):
    regr = RandomForestRegressor(max_depth=4, random_state=0, n_jobs=-1)
    regr.fit(X_train, y_train)

    return regr


def model_cross_validation(model, 
                            X_train, y_train,
                            max_depth = [4,5],
                            min_samples_leaf= [50,75], 
                            n_estimators = [100,150]):
    params = {
    'max_depth': max_depth,
    'min_samples_leaf': min_samples_leaf,
    'n_estimators': n_estimators
}

    grid_search = GridSearchCV(estimator=model,
                           param_grid=params,
                           cv = 3,
                           n_jobs=-1, verbose=1, scoring="r2")
    
    grid_search.fit(X_train, y_train)

    print(grid_search.best_score_)

    return grid_search.best_estimator_


def trainging_pipeline_run(X_train, y_train):
    # X_train, X_test, y_train, y_test = data_split(df)
    model = model_init_and_fit(X_train, y_train)
    best_model = model_cross_validation(model, 
                            X_train, y_train)
    return best_model

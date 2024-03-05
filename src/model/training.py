from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from utils import load_config

config = load_config()

# Data split configs
TEST_SIZE = config["data_split"]["test_size"]
RANDOM_STATE = config["data_split"]["random_state"]

# Model initialization and fit configs
MAX_DEPTH = config["model_initialization"]["max_depth"]
RANDOM_STATE = config["model_initialization"]["random_state"]
N_JOBS = config["model_initialization"]["n_jobs"]

# Model cross validation configs
CV = config["model_cross_validation"]["cv"]
N_JOBS_CV = config["model_cross_validation"]["n_jobs"]
VERBOSE = config["model_cross_validation"]["verbose"]
SCORING = config["model_cross_validation"]["scoring"]

# Cross validtion search grid
CV_MAX_DEPTH = config["model_cross_validation"]["param_grid"]["max_depth"]
CV_MIN_SAMPLES_LEAF = config["model_cross_validation"]["param_grid"]["min_samples_leaf"]
CV_N_ESTIMATORS = config["model_cross_validation"]["param_grid"]["n_estimators"]


def data_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = df[
        [
            "dist_to_restaurant",
            "Hdist_to_restaurant",
            "avg_Hdist_to_restaurants",
            "date_day_number",
            "restaurant_id",
            "Five_Clusters_embedding",
            "h3_index",
            "date_hour_number",
            "restaurants_per_index",
        ]
    ]
    y = df[["orders_busyness_by_h3_hour"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def model_init_and_fit(
    X_train, y_train, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=N_JOBS
):
    regr = RandomForestRegressor(
        max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
    )
    regr.fit(X_train, y_train)

    return regr


def model_cross_validation(
    model,
    X_train,
    y_train,
    max_depth=CV_MAX_DEPTH,
    min_samples_leaf=CV_MIN_SAMPLES_LEAF,
    n_estimators=CV_N_ESTIMATORS,
    cv=CV,
    n_jobs=N_JOBS_CV,
    verbose=VERBOSE,
    scoring=SCORING,
):
    params = {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_estimators": n_estimators,
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring=scoring,
    )

    grid_search.fit(X_train, y_train)
    print(grid_search.best_score_)
    return grid_search.best_estimator_


def trainging_pipeline_run(X_train, y_train):
    model = model_init_and_fit(X_train, y_train)
    best_model = model_cross_validation(model, X_train, y_train)
    return best_model

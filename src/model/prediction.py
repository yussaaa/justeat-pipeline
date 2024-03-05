from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def model_predict(model, X_test):

    y_pred = model.predict(X_test)
    return y_pred


def model_evaluate(y_test, y_pred):

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2

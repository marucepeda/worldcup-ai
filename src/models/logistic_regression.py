"""Regresión logística multiclase sobre el dataset completo de features."""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.common import load_dataset, report, time_based_split


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = time_based_split(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report("Logistic Regression", y_test, predictions)


if __name__ == "__main__":
    main()

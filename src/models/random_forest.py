"""
Random Forest.

No necesita escalado de features (a diferencia de la regresión
logística), y puede capturar relaciones no lineales entre las
variables. Se compara contra los modelos lineales para ver si la
complejidad adicional realmente se traduce en mejor desempeño.
"""

from sklearn.ensemble import RandomForestClassifier

from src.models.common import load_dataset, report, time_based_split

N_ESTIMATORS = 200
RANDOM_STATE = 42


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = time_based_split(df)

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report("Random Forest", y_test, predictions)


if __name__ == "__main__":
    main()

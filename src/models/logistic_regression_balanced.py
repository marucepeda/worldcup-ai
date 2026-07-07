"""
Regresión logística con class_weight="balanced".

El target tiene clases desbalanceadas (el equipo local gana con más
frecuencia que las otras dos opciones). Esta variante le da más peso
a los empates y a las victorias visitantes durante el entrenamiento.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.common import load_dataset, report, time_based_split


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = time_based_split(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report("Logistic Regression (Balanced)", y_test, predictions)


if __name__ == "__main__":
    main()

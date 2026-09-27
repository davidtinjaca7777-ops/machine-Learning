from flask import request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def init_randomforest_routes(app):
    @app.route("/random_forest_app", methods=["GET", "POST"])
    def random_forest_app():
        data = pd.read_csv("static/dataset_random_forest.csv")
        X = data[["edad","ingreso_mensual","visitas_web_mes","tiempo_sitio_min","compras_previas","descuento_usado"]]
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred).tolist()

        prediction = None
        if request.method == "POST":
            edad = int(request.form["edad"])
            ingreso = int(request.form["ingreso_mensual"])
            visitas = int(request.form["visitas_web_mes"])
            tiempo = float(request.form["tiempo_sitio_min"])
            compras = int(request.form["compras_previas"])
            descuento = int(request.form["descuento_usado"])

            prediction = model.predict([[edad, ingreso, visitas, tiempo, compras, descuento]])[0]
            prediction = "Compra (1)" if prediction == 1 else "No compra (0)"

        return render_template(
            "random_forest_app.html",
            prediction=prediction,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            confusion=confusion
        )

from flask import request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def init_logistic_routes(app):
    @app.route("/logistic_regression_app", methods=["GET", "POST"])
    def logistic_regression_app():
        data = pd.read_csv("static/dataset_regresion_logistica.csv")

        independent_vars = ["edad", "ingreso_mensual", "visitas_web_mes", "tiempo_sitio_min", "compras_previas", "descuento_usado"]
        target_var = "target"

        X = data[independent_vars]
        y = data[target_var]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        prediction = None
        if request.method == "POST":
            edad = int(request.form["edad"])
            ingreso = int(request.form["ingreso_mensual"])
            visitas = int(request.form["visitas_web_mes"])
            tiempo = float(request.form["tiempo_sitio_min"])
            compras = int(request.form["compras_previas"])
            descuento = int(request.form["descuento_usado"])

            prediction = model.predict([[edad, ingreso, visitas, tiempo, compras, descuento]])[0]

        return render_template(
            "logistic_app.html",
            num_records=len(data),
            independent_vars=independent_vars,
            target_var=target_var,
            class_meanings={0: "No Purchase", 1: "Purchase"},
            train_size=len(X_train),
            test_size=len(X_test),
            accuracy=round(accuracy, 3),
            precision=round(precision, 3),
            recall=round(recall, 3),
            confusion=confusion.tolist(),
            prediction=prediction
        )

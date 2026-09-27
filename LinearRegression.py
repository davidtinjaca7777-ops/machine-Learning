from flask import request, render_template
from LinearModel import predecir_ingresos  

def init_linear_routes(app):
    @app.route("/linear_regression_app", methods=["GET", "POST"])
    def linear_regression_app():
        prediction = None
        if request.method == "POST":
            clientes = int(request.form["clientes"])
            prediction = predecir_ingresos(clientes)

        return render_template("ingresos.html", prediccion=prediction)

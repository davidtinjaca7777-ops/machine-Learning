from flask import Flask, render_template

from Kmeans import init_kmeans_routes
from RandomForest import init_randomforest_routes
from LinearRegression import init_linear_routes
from LogisticRegression import init_logistic_routes

app = Flask(__name__)

@app.route("/machine")
def machine():
    return render_template("machine.html")


@app.route("/case1")
def case1():
    return render_template("case1.html")

@app.route("/case2")
def case2():
    return render_template("case2.html")

@app.route("/case3")
def case3():
    return render_template("case3.html")

@app.route("/case4")
def case4():
    return render_template("case4.html")

@app.route("/linear_regression_concepts")
def linear_regression_concepts():
    return render_template("linear_regression_concepts.html")


@app.route("/logistic_regression_concepts")
def logistic_regression_concepts():
    return render_template("logistic_regression_concepts.html")

@app.route("/random_forest_concepts")
def random_forest_concepts():
    return render_template("random_forest_concepts.html")

@app.route("/decision_threshold")
def decision_threshold():
    return render_template("decision_threshold.html")

@app.route("/comparison_algorithms")
def comparison_algorithms():
    return render_template("comparison_algorithms.html")


@app.route("/kmeans_concepts")
def kmeans_concepts():
    return render_template("kmeans_concepts.html")

import pandas as pd

@app.route("/kmeans_manual")
def kmeans_manual():
    df = pd.read_excel("static/KMeans_Iteracion_1_Estudiantes (1).xlsx")
    table = df.to_html(classes="table table-striped", index=False)
    return render_template("kmeans_manual.html", table=table)


@app.route("/")
def index():
    return render_template("menu.html")

# Registrar las apps de cada modelo
init_kmeans_routes(app)
init_randomforest_routes(app)
init_linear_routes(app)
init_logistic_routes(app)

if __name__ == "__main__":
    app.run(debug=True)

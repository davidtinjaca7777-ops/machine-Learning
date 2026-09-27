from flask import request, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def init_kmeans_routes(app):
    @app.route("/kmeans_app", methods=["GET", "POST"])
    def kmeans_app():
        # Cargar dataset
        data = pd.read_csv("static/Estudiantes.csv")

        # Usa las columnas correctas
        X = data[["Study_Hours", "Calification"]]

        # Entrenar modelo K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)

        # Agregar columna de cluster al dataset
        data["Cluster"] = labels

        # Calcular centroides
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=["Study_Hours", "Calification"])
        centroids["Cluster"] = centroids.index

        # Métricas
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)

        prediction = None
        horas = None
        nota = None

        if request.method == "POST":
            horas = float(request.form["study_hours"])
            nota = float(request.form["grade"])
            prediction = kmeans.predict([[horas, nota]])[0]

        # Guardar gráfico con clusters, centroides y nuevo punto si existe
        plt.figure(figsize=(6,4))
        plt.scatter(X["Study_Hours"], X["Calification"], c=labels, cmap="viridis", label="Students")
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c="red", marker="x", s=100, label="Centroids")

        if prediction is not None:
            plt.scatter(horas, nota, c="black", marker="o", s=120, label="New Student")

        plt.xlabel("Study Hours")
        plt.ylabel("Calification")
        plt.title("K-Means Clustering")
        plt.legend()
        plt.savefig("static/kmeans_plot.png")
        plt.close()

        # Convertir tabla a HTML
        table_html = data.to_html(classes="table table-bordered table-striped", index=False)

        return render_template(
            "kmeans_app.html",
            prediction=prediction,
            inertia=round(inertia, 2),
            silhouette=round(silhouette, 3),
            table=table_html,
            centroids=centroids.to_dict(orient="records")
        )

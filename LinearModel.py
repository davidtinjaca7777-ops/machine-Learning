import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {
    "clientes": [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150],
    "ingresos": [950, 2100, 3100, 4050, 5200, 6100, 6900, 8200, 10200, 12500, 15500]
}

df = pd.DataFrame(data)

x = df[["clientes"]]
y = df["ingresos"]

modelo = LinearRegression()
modelo.fit(x, y)

def predecir_ingresos(clientes):
    prediccion = modelo.predict([[clientes]])[0]

    
    plt.scatter(df["clientes"], df["ingresos"], color="blue", label="Datos reales")
    plt.plot(df["clientes"], modelo.predict(x), color="red", label="Regresión")
    plt.scatter([clientes], [prediccion], color="green", marker="x", s=100, label="Predicción")

    plt.xlabel("Clientes")
    plt.ylabel("Ingresos")
    plt.title("Predicción de ingresos")
    plt.legend()
    plt.savefig("static/grafico.png")
    plt.close()

    return prediccion


import numpy as np
import pandas as pd
import scipy.stats  as stats
import seaborn as sns
from scipy.stats import f
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from scipy.stats import studentized_range
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import tukey_hsd
import itertools
from itertools import combinations
from tabulate import tabulate

fig, ax = plt.subplots(1, 1)


#Listas para almacenar los datos ingresados
oxido_nitrosoy = []
humedadx1 = []
temperaturax2 = []
presionx3 = []

#Funcion para pedir y tratar las listas
def ingresar_listas(nombre):
    while True:
        valores = input(f"Ingrese los valores de {nombre} separados por comas: ").strip()
        if valores:
            try:
                lista_numeros = [float(valor) for valor in valores.split(",")]
                return lista_numeros
            except ValueError:
                print("⚠️ Error: Debe ingresar solo números separados por comas.")
        else:
            print("⚠️ Este campo es obligatorio. Intente de nuevo.")

print("\n🔹 Ingrese los valores fila por fila.")


#Consola de usuario
y = ingresar_listas("y (Óxido nitroso)")
x1 = ingresar_listas("x1 (Humedad)")
x2 = ingresar_listas("x2 (Temperatura)")
x3 = ingresar_listas("x3 (Presión)")


#Agregar los valores a las listas
oxido_nitrosoy.extend(y)
humedadx1.extend(x1)
temperaturax2.extend(x2)
presionx3.extend(x3)


#Determinar automáticamente el número de columnas (t)
columnas = [oxido_nitrosoy, humedadx1, temperaturax2, presionx3]
t = len(columnas)

#Solicitar el nivel de significancia después del ingreso de datos
while True:
    try:
        nivel_significancia = float(input("\n🔹 Ingrese el nivel de significancia (ejemplo: 95 para 95%): "))
        if 0 < nivel_significancia <= 100:
            nivel_significancia /= 100  # Convertir a decimal (ejemplo: 95 → 0.95)
            break
        else:
            print("⚠️ Error: Debe ingresar un valor entre 0 y 100.")
    except ValueError:
        print("⚠️ Error: Debe ingresar un número válido.")

print(f"\n✅ Nivel de significancia ingresado: {nivel_significancia}")

print("\n✅ Datos ingresados correctamente. Próximamente se mostrarán los resultados.")


#Verificar que todas las listas tengan la misma cantidad de elementos
min_length = min(len(y), len(x1), len(x2), len(x3))

#Ajustar las listas para que tengan el mismo tamaño
y = y[:min_length]
x1 = x1[:min_length]
x2 = x2[:min_length]
x3 = x3[:min_length]

#crar tabla de datos ingresados
df = pd.DataFrame({
    "Óxido Nitroso (y)": y,
    "Humedad (x1)": x1,
    "Temperatura (x2)": x2,
    "Presión (x3)": x3
})

print(df)

#Σxt sumatorias de los elementos de cada columna
Σxt_oxido_nitrosoy = sum(oxido_nitrosoy)
Σxt_humedadx1  = sum(humedadx1)
Σxt_temperaturax2  = sum(temperaturax2)
Σxt_presionx3  = sum(presionx3)

print(f"Σyt: {round(Σxt_oxido_nitrosoy, 4)}, Σx1t humedadx1: {round(Σxt_humedadx1, 4)} Σx2t humedadx1: {round(Σxt_temperaturax2, 4)}, Σx3t: {round(Σxt_presionx3, 4)} ")

Σxt_4col = sum([Σxt_oxido_nitrosoy,Σxt_humedadx1, Σxt_temperaturax2, Σxt_presionx3])

print ("Σxt", round(Σxt_4col, 4))


#Sumatorias de los elementos elevados al cuadrado
Σxt2_oxido_nitrosoy = sum([elemento ** 2 for elemento in oxido_nitrosoy])
Σxt2_humedadx1 = sum([elemento ** 2 for elemento in humedadx1])
Σxt2_temperaturax2 = sum([elemento ** 2 for elemento in temperaturax2])
Σxt2_presionx3 = sum([elemento ** 2 for elemento in presionx3])

print(f" Σyt²: {round(Σxt2_oxido_nitrosoy, 4)}, Σx1t humedadx1: {round(Σxt2_humedadx1, 4)}, Σx2t² temperaturax2: {round(Σxt2_temperaturax2, 4)}, Σx3t² presionx3: {round(Σxt2_presionx3, 4)}")


Σxt2_4col = sum([Σxt2_oxido_nitrosoy,Σxt2_humedadx1, Σxt2_temperaturax2, Σxt2_presionx3])

print ("Σxt²: ", round(Σxt2_4col, 4))
  

#Sumatorias de las sumas de los lementos de las muestras elevado al cuadrado
Σxtcua_oxido_nitrosoy =  sum(oxido_nitrosoy)  ** 2
Σxtcua_humedadx1  = sum(humedadx1) ** 2
Σxtcua_temperaturax2  =  sum(temperaturax2) ** 2
Σxtcua_presionx3  = sum(presionx3) ** 2

print(f" (Σyt)²: {round(Σxtcua_oxido_nitrosoy, 4)}, Σx1t humedadx1: {round(Σxtcua_humedadx1, 4)}, Σx2t² temperaturax2: {round(Σxtcua_temperaturax2, 4)}, Σx3t² presionx3: {round(Σxtcua_presionx3, 4)}")

Σxtcua_4col = sum([Σxtcua_oxido_nitrosoy,Σxtcua_humedadx1, Σxtcua_temperaturax2, Σxtcua_presionx3])

print ("(Σxt)²: ", round(Σxtcua_4col, 4))

#Sumatoria de la cantidad de elementos por columna
nt_oxido_nitrosoy = len(oxido_nitrosoy)
nt_humedadx1 =  len(humedadx1)
nt_temperaturax2 = len(temperaturax2)
nt_presionx3 = len(presionx3)

print(f" nty: {nt_oxido_nitrosoy}, ntx1: {round(nt_humedadx1)}, ntx2: {nt_temperaturax2}, ntx3: {nt_presionx3}")

Σnt_4col = sum([nt_oxido_nitrosoy,nt_humedadx1, nt_temperaturax2, nt_presionx3])

print ("Σnt: ", Σnt_4col)

#Sumatorias de los elementos elevados al cuadrado divido entre n (cantidad de elementos)
ΣxtcuaN_oxido_nitrosoy =  Σxtcua_oxido_nitrosoy /nt_oxido_nitrosoy
ΣxtcuaN_humedadx1  = Σxtcua_humedadx1 / nt_humedadx1
ΣxtcuaN_temperaturax2  = Σxtcua_temperaturax2 / nt_temperaturax2
ΣxtcuaN_presionx3  = Σxtcua_presionx3 / nt_presionx3

print(f" (Σyt)²/n: {round(ΣxtcuaN_oxido_nitrosoy, 4)}, (Σx1t)²/n: {round(ΣxtcuaN_humedadx1, 4)}, (Σx2t)²/n: { round(ΣxtcuaN_temperaturax2, 4)}, (Σx3t)²/n: {round(ΣxtcuaN_presionx3, 4)}")


ΣxtcuaN_4col = sum([ΣxtcuaN_oxido_nitrosoy, ΣxtcuaN_humedadx1, ΣxtcuaN_temperaturax2, ΣxtcuaN_presionx3])
print (" (Σxt)²/n: ", ΣxtcuaN_4col)

#Sumatoria de las media aritmeticas
x_oxido_nitrosoy = Σxt_oxido_nitrosoy / nt_oxido_nitrosoy
x_humedadx1 = Σxt_humedadx1 / nt_humedadx1
x_temperaturax2 = Σxt_temperaturax2 / nt_temperaturax2
x_presionx3 =  Σxt_presionx3 / nt_presionx3

print(f" x̅y: {round(x_oxido_nitrosoy, 4)}, x̅x1: {round(x_humedadx1, 4)}, x̅x2: {round(x_temperaturax2)}, x̅x3: {round(x_presionx3)}")

x_4col = sum([x_oxido_nitrosoy, x_humedadx1, x_temperaturax2, x_presionx3])
print ("x̅: ", round(x_4col))

#Calcular nivel de significancia
alfa = 1 - nivel_significancia
print(f"α (alfa):" , round(alfa, 4))

#grados de libertad del tratamiento
gl_tratamiento =   t -1
print(f"gl(tratamiento):" , gl_tratamiento)

#grados de libertad del error
gl_error = Σnt_4col - t
print(f"gl(error):" , round(gl_error, 4))

#Factor de correcion
c =  Σxt_4col ** 2  / Σnt_4col
print(f"Factor de correcion (C):" , round(c, 4))

#Suma Total de Cuadradados
sct = Σxt2_4col - c
print(f"Suma Total de Cuadrados (SCT):" , round(sct, 4))

#Suma Cuadradada de Tratamiento
sctr = ΣxtcuaN_4col - c
print(f"Suma Cuadradada de Tratamiento (SCTR):" , round(sctr, 4))

#Suma Cuadradada de error
sce = Σxt2_4col - ΣxtcuaN_4col
print(f"Suma Cuadradada de Error (SCE):" , round(sce, 4))

#n - 1
nmenos1 = Σnt_4col - 1

#MCTR
mctr = sctr / gl_tratamiento

#MCE
mce = sce / gl_error

#F(RV) Fisher razón de variacion
f_rv = mctr / mce


#Crear DataFrame con pandas de la fuente de variacion

fuente_variacion = pd.DataFrame({
    "Fuentes de variacion": ["Tratamiento", "Error", "Total"],
    "SC": [round(sctr, 4), round(sce, 4), round(sct, 4)],
    "gl": [round(gl_tratamiento, 4), round(gl_error, 4), nmenos1],
    "MC": [round(mctr, 4), round(mce, 4), None],  
    "F(RV)": [round(f_rv, 4), None, None]  
})


print(fuente_variacion)



#Buscar F tabulada 
Ftab = stats.f.ppf(1 - alfa, gl_tratamiento, gl_error)
print(f"F tabulada: {round(Ftab, 4)}")

#Comparación y decision
print("[Si  Fcal > Ftab = RR]" , "Fcalc < Ftab = RA")
if f_rv > Ftab:
    decision = "Rechazar H₀ (Existe diferencia significativa)"
else:
    decision = "Aceptar H₀ (No hay diferencia significativa)"

print(f"Decisión: {decision}")

#Prueba DHS
# Número de grupos (columnas)
num_grupos = t

# Valor crítico q para HSD
q = studentized_range.ppf( 1 - alfa, num_grupos, gl_error)

# Calcular DHS
hsd = q * np.sqrt(mce / nt_oxido_nitrosoy)  # Usamos uno de los n's (asumido igual para todos)
print()
print(f"Valor crítico q: {q}")
print(f"Diferencia Honestamente Significativa (HSD): {hsd}")



#Tukey 

medias = [x_oxido_nitrosoy, x_humedadx1, x_temperaturax2, x_presionx3]

# Definir las medias de cada grupo
medias = {
    "Óxido Nitroso": x_oxido_nitrosoy,
    "Humedad": x_humedadx1,
    "Temperatura": x_temperaturax2,
    "Presión": x_presionx3
}


# Lista de pares para comparar
pares = [
    ("Óxido Nitroso", "Humedad"),
    ("Óxido Nitroso", "Temperatura"),
    ("Óxido Nitroso", "Presión"),
    ("Humedad", "Temperatura"),
    ("Humedad", "Presión"),
    ("Temperatura", "Presión")
]


ind_vars = df.columns

# Construir el diccionario de medias a partir de los nombres reales del DataFrame:
medias = {col: df[col].mean() for col in ind_vars}
pares = list(combinations(ind_vars, 2))

independientes = []

print("Comparación de Medias - Prueba de Tukey\n")
print(f"{'Grupo 1':<15}{'Grupo 2':<15}{'Diferencia':<15}{'DHS':<10}{'Independencia'}")
print("-" * 80)



for g1, g2 in pares:
    diff = medias[g1] - medias[g2]

    if diff > hsd:  
        estado = "Independiente"
        independientes.append((g1, g2))
    else:
        estado = "Dependiente"

    # Imprimir resultado en la tabla
    print(f"{g1:<20}{g2:<20}{diff:<15.4f}{hsd:<10.4f}{estado}")



    
if independientes:
    print("\nCálculo de correlación y regresión para pares independientes:\n")
    for g1, g2 in independientes:
        print(f"Análisis entre {g1} y {g2}:")
        x = df[g1]
        y = df[g2]
        
        # Crear una tabla (DataFrame) del par
        tab = pd.DataFrame({"x": x, 
                            "y": y})
        print("Tabla de datos:")
        print(tab)
        
        # Sumatorias
        sum_x = x.sum()
        sum_y = y.sum()
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_xy = np.sum(x * y)
        
        print(f"Σx: {sum_x:.4f}, Σy: {sum_y:.4f}")
        print(f"Σx²: {sum_x2:.4f}, Σy²: {sum_y2:.4f}, Σxy: {sum_xy:.4f}")
        
        # Calcular correlación 
        n = len(x)

        # Sumatorias
        Σx = sum(x)
        Σy = sum(y)
        Σxy = sum(xi * yi for xi, yi in zip(x, y))
        Σx2 = sum(xi ** 2 for xi in x)
        Σy2 = sum(yi ** 2 for yi in y)

        # Media de x y y
        x̄ = Σx / n
        ȳ = Σy / n

        # Coeficiente de correlación de Pearson
        numerador_r = (n * Σxy) - (Σx * Σy)
        denominador_r = ((n * Σx2 - Σx ** 2) * (n * Σy2 - Σy ** 2)) ** 0.5
        r = numerador_r / denominador_r if denominador_r != 0 else 0

        print(f"Coeficiente de correlación: {r:.4f}")

        # Cálculo de la pendiente (b) y la intersección (a)
        b = numerador_r / (n * Σx2 - Σx ** 2) if (n * Σx2 - Σx ** 2) != 0 else 0
        a = ȳ - (b * x̄)

        print(f"Ecuación de regresión: {g2} = {a:.4f} + {b:.4f} * {g1}\n")


        # Crear el diagrama de dispersión con la recta de regresión
        #plt.figure(figsize=(8, 6))  
        plt.scatter(x, y, color='blue', label='Datos')  # Puntos de dispersión
        plt.plot(x, a + b * x, color='red', label=f'Recta de regresión: {g2} = {a:.4f} + {b:.4f} * {g1}')  # Recta de regresión
        plt.title(f"Diagrama de dispersión y regresión lineal: {g1} vs {g2}")  # Título
        plt.xlabel(f'{g1}')
        plt.ylabel(f'{g2}')
        plt.legend() 
        plt.grid(True) 
        plt.show()  


        

     
else:
    print("No se encontraron pares independientes (diferencia > DHS).")

#Tabla de Regresion multiple
dfmultiple = pd.DataFrame({
    "Óxido Nitroso (y)": y,
    "Humedad (x1)": x1,
    "Temperatura (x2)": x2,
    "Presión (x3)": x3,
    "y^2": np.square(y),
    "x1^2": np.square(x1),
    "x2^2": np.square(x2),
    "x3^2": np.square(x3),
    "y*x1": np.multiply(y, x1),
    "y*x2": np.multiply(y, x2),
    "y*x3": np.multiply(y, x3),
    "x1*x2": np.multiply(x1, x2),
    "x2*x3": np.multiply(x2, x3),
    "x1*x3": np.multiply(x1, x3)
})

sumatorias = dfmultiple.sum()
dfmultiple.loc["-------------"] = ["-" * 10] * dfmultiple.shape[1]
dfmultiple.loc["Σ"] = sumatorias


# Mostrar el DataFrame con las sumatorias
print("\nTabla de Contingencia con Datos Calculados:")
print(dfmultiple)

"""
# Mostrar los resultados de las sumatorias en caso de que la tabla se resuma
print("\n****Resultados de sumatorias en caso de que la tabla se resuma****")
print(f"Σyt: {round(sumatorias['Óxido Nitroso (y)'], 4)}")
print(f"Σx1t (Humedad): {round(sumatorias['Humedad (x1)'], 4)}")
print(f"Σx2t (Temperatura): {round(sumatorias['Temperatura (x2)'], 4)}")
print(f"Σx3t (Presión): {round(sumatorias['Presión (x3)'], 4)}")
print(f"Σy^2: {round(sumatorias['y^2'], 4)}")
print(f"Σx1^2: {round(sumatorias['x1^2'], 4)}")
print(f"Σx2^2: {round(sumatorias['x2^2'], 4)}")
print(f"Σx3^2: {round(sumatorias['x3^2'], 4)}")
print(f"Σy*x1: {round(sumatorias['y*x1'], 4)}")
print(f"Σy*x2: {round(sumatorias['y*x2'], 4)}")
print(f"Σy*x3: {round(sumatorias['y*x3'], 4)}")
print(f"Σx1*x2: {round(sumatorias['x1*x2'], 4)}")
print(f"Σx2*x3: {round(sumatorias['x2*x3'], 4)}")
print(f"Σx1*x3: {round(sumatorias['x1*x3'], 4)}")"""

def gauss_jordan(A, B):
    AB = np.hstack([A, B.reshape(-1, 1)])  # Matriz ampliada [A|B]
    n = len(B)
    #hstack apila matrices en secuencia horizontalmente (columna por columna).
    for i in range(n):
        AB[i] = AB[i] / AB[i, i]
        
        # Hacer ceros en las demás filas
        for j in range(n):
            if i != j:
                AB[j] = AB[j] - AB[j, i] * AB[i]
    return AB 


def determinar_regresion(dfmultiple):
    try:
        sumatorias = dfmultiple.loc["Σ"]
        n = nt_humedadx1  # Considerando que todas tienen la misma longitud

        # Construir la matriz A
        A = np.array([
            [n, sumatorias["Temperatura (x2)"], sumatorias["Presión (x3)"]],
            [sumatorias["Temperatura (x2)"], sumatorias["x2^2"], sumatorias["x2*x3"]],
            [sumatorias["Presión (x3)"], sumatorias["x2*x3"], sumatorias["x3^2"]]
        ])

        # Verificar si la matriz A es invertible
        det_A = np.linalg.det(A)
        if np.isclose(det_A, 0):
            raise ValueError("La matriz A no es invertible (det(A) = 0). El sistema no tiene solución única.")

        # Vector B
        B = np.array([
            sumatorias["Humedad (x1)"],
            sumatorias["x1*x2"],
            sumatorias["x1*x3"]
        ])

        # Mostrar la matriz ampliada [A|B]
        print("\nMatriz ampliada [A|B]:")
        print(tabulate(np.hstack([A, B.reshape(-1, 1)]), headers=["B0", "B1", "B2", "B"], tablefmt='grid', floatfmt='.4f'))

        # Asignamos a una variable el resultado del metodo gauss con los valores de A y B
        matriz_final = gauss_jordan(A, B)

        # Mostrar la matriz final
        print("\nMatriz final:")
        print(tabulate(matriz_final, headers=["B0", "B1", "B2", "B"], tablefmt='grid', floatfmt='.4f'))

        # Extraer los resultados de la última columna
        resultados = matriz_final[:, -1]

        # Mostrar resultados
        print("\nResultados:")
        print(f"β0 = {resultados[0]:.4f}")
        print(f"β1 = {resultados[1]:.4f}")
        print(f"β2 = {resultados[2]:.4f}")

        #Recta
        print("Recta de regresión múltiple")
        print("y = β + β1 + x2")
        print(f" y = β0{resultados[0]:.4f} + β1{resultados[1]:.4f} + β2{resultados[2]:.4f} x2")

        return resultados
    except Exception as e:
        print(f"Error al calcular la regresión: {e}")
        return None


# Calcular los coeficientes de regresión
coeficientes = determinar_regresion(dfmultiple)
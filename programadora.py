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
import traceback


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

print("\n🔹 Ingrese los valores de las variables.")

# Ingresar variable dependiente (y)
y_valores = ingresar_listas("y")

# Diccionario para almacenar los datos
datos = {"y": y_valores}

# Ingresar variables independientes dinámicamente
contador_columnas = 0
while True:
    if contador_columnas >= 1:  # Preguntar si desea continuar a partir de la segunda variable independiente
        continuar = input("\n¿Desea ingresar otra columna? (s/n): ").strip().lower()
        if continuar != 's':
            break  # Salir del bucle si el usuario no desea continuar

    # Solicitar el nombre de la nueva variable independiente
    while True:
        nombre_columna = input("\nIngrese el nombre de la nueva variable independiente: ").strip()
        if nombre_columna and nombre_columna not in datos:
            break  # Salir del bucle si el nombre no está vacío ni repetido
        print("⚠️ El nombre de la columna no puede estar vacío o repetido. Intente de nuevo.")

    # Solicitar los valores de la nueva variable independiente
    valores = ingresar_listas(nombre_columna)
    datos[nombre_columna] = valores
    contador_columnas += 1

# Ajustar todas las listas al mismo tamaño
min_length = min(len(col) for col in datos.values())
for key in datos:
    datos[key] = datos[key][:min_length]

# Crear y mostrar el DataFrame
df = pd.DataFrame(datos)

print("\n✅ Datos ingresados correctamente. Aquí está la tabla final:")
print(df)



# Calcular el número de columnas dinámicamente
t = len(datos)


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



# 🔹 **Cálculo de Sumatorias de forma dinámica**
sumatorias = {col: sum(datos[col]) for col in datos}  # Σxt: Suma de cada columna
sumatorias_cuadradas = {col: sum(x ** 2 for x in datos[col]) for col in datos}  # Σxt²: Suma de cada elemento al cuadrado
sumatorias_total_cuadrado = {col: sum(datos[col]) ** 2 for col in datos}  # (Σxt)²: Suma total elevada al cuadrado
cantidad_elementos = {col: len(datos[col]) for col in datos}  # n: Cantidad de elementos por columna
sumatorias_divididas_n = {col: sumatorias_total_cuadrado[col] / cantidad_elementos[col] for col in datos}  # (Σxt)²/n

# 🔹 **Mostrar resultados**
print("\n🔹 **Sumatorias de cada columna**")
for col, suma in sumatorias.items():
    print(f"Σ{col} = {round(suma, 4)}")

print("\n🔹 **Sumatorias de los elementos al cuadrado**")
for col, suma in sumatorias_cuadradas.items():
    print(f"Σ{col}² = {round(suma, 4)}")

print("\n🔹 **Sumatorias de la suma de los elementos elevados al cuadrado**")
for col, suma in sumatorias_total_cuadrado.items():
    print(f"(Σ{col})² = {round(suma, 4)}")

print("\n🔹 **Cantidad de elementos por columna**")
for col, cantidad in cantidad_elementos.items():
    print(f"n{col} = {cantidad}")

print("\n🔹 **Sumatorias de los elementos elevados al cuadrado dividido entre n**")
for col, suma in sumatorias_divididas_n.items():
    print(f"(Σ{col})²/n = {round(suma, 4)}")

# 🔹 **Sumatoria total de todas las columnas**
Σxt_total = sum(sumatorias.values())
Σxt2_total = sum(sumatorias_cuadradas.values())
Σxtcua_total = sum(sumatorias_total_cuadrado.values())
Σnt_total = sum(cantidad_elementos.values())
ΣxtcuaN_total = sum(sumatorias_divididas_n.values())

print("\n**Resultados Totales**")
print(f"Σxt total: {round(Σxt_total, 4)}")
print(f"Σxt² total: {round(Σxt2_total, 4)}")
print(f"(Σxt)² total: {round(Σxtcua_total, 4)}")
print(f"Σnt total: {Σnt_total}")
print(f"(Σxt)²/n total: {round(ΣxtcuaN_total, 4)}")



#Cálculo de la Suma Total de Cuadrados (SCT)
sct = Σxt2_total - Σxtcua_total / Σnt_total
print(f"Suma Total de Cuadrados (SCT): {round(sct, 4)}")

# Cálculo de la Suma Cuadrada de Tratamiento (SCTR)
sctr = ΣxtcuaN_total - Σxtcua_total / Σnt_total
print(f"Suma Cuadrada de Tratamiento (SCTR): {round(sctr, 4)}")

#Cálculo de la Suma Cuadrada del Error (SCE)
sce = Σxt2_total - ΣxtcuaN_total
print(f"Suma Cuadrada del Error (SCE): {round(sce, 4)}")

# Grados de libertad
gl_total = Σnt_total - 1
gl_tratamiento = len(datos) - 1  # Número de variables independientes
gl_error = gl_total - gl_tratamiento

# Cálculo de MCTR (Media Cuadrática del Tratamiento)
mctr = sctr / gl_tratamiento
print(f"Media Cuadrática del Tratamiento (MCTR): {round(mctr, 4)}")

#Cálculo de MCE (Media Cuadrática del Error)
mce = sce / gl_error
print(f"Media Cuadrática del Error (MCE): {round(mce, 4)}")

#Cálculo del F (Razón de Variación de Fisher)
f_rv = mctr / mce
print(f"F (Razón de Variación de Fisher): {round(f_rv, 4)}")

#n - 1
nmenos1 = Σxt_total - 1

#Nivel de significancia
alfa = 1 - nivel_significancia

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
print("[Si  Fcal > Ftab = RR] \n calc < Ftab = RA")
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
cantidad_total_elementos = cantidad_elementos["y"]  # Asumiendo que todas tienen la misma longitud
hsd = q * np.sqrt(mce / cantidad_total_elementos)
print(f"Valor crítico q: {q}")
print(f"Diferencia Honestamente Significativa (HSD): {hsd}")

#Tukey 
# Calcular las medias de cada columna
medias = {col: df[col].mean() for col in df.columns}

# Generar todos los pares posibles de combinaciones de columnas
pares = list(combinations(df.columns, 2))

# Lista para almacenar pares independientes
independientes = []

# Mostrar encabezado de la tabla de Tukey
print("Comparación de Medias - Prueba de Tukey\n")
print(f"{'Grupo 1':<20}{'Grupo 2':<20}{'Diferencia':<15}{'DHS':<10}{'Independencia'}")
print("-" * 80)

# Comparar cada par de medias
for g1, g2 in pares:
    diff = medias[g1] - medias[g2]  # Diferencia entre las medias
    if diff > hsd:  # Si la diferencia es mayor que HSD, son independientes
        estado = "Independiente"
        independientes.append((g1, g2))
    else:
        estado = "Dependiente"

    # Imprimir resultado en la tabla
    print(f"{g1:<20}{g2:<20}{diff:<15.4f}{hsd:<10.4f}{estado}")

#Correlación y regresión de variables indpendientes
if independientes:
    print("\nCálculo de correlación y regresión para pares independientes:\n")
    for g1, g2 in independientes:
        print(f"Análisis entre {g1} y {g2}:")
        x = df[g1]
        y = df[g2]
        
        # Crear una tabla (DataFrame) del par
        tab = pd.DataFrame({"x": x, "y": y})
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

        # Cálculo de las interssciones entre a y b
        b = numerador_r / (n * Σx2 - Σx ** 2) if (n * Σx2 - Σx ** 2) != 0 else 0
        a = ȳ - (b * x̄)

        print(f"Ecuación de regresión: {g2} = {a:.4f} + {b:.4f} * {g1}\n")

        # Crear el diagrama de dispersión con la recta de regresión
       # plt.figure(figsize=(8, 6))  
        plt.scatter(x, y, color='blue', label='Datos') 
        plt.plot(x, a + b * x, color='red', label=f'Recta de regresión: {g2} = {a:.4f} + {b:.4f} * {g1}')  # Recta de regresión
        plt.title(f"Diagrama de dispersión y regresión lineal: {g1} vs {g2}")  # Título
        plt.xlabel(g1)  
        plt.ylabel(g2) 
        plt.legend()
        plt.grid(True) 
        plt.show()
else:
    print("No se encontraron pares independientes (diferencia > DHS).")



#Tabla de Regresion multiple
def crear_tabla_regresion(df):
    # Crear un nuevo DataFrame 
    df_regresion = df.copy()

    # Calcular las columnas adicionales (cuadrados y productos)
    for col in df.columns:
        df_regresion[f"{col}^2"] = df[col] ** 2  # Cuadrados de cada columna

    # Calcular multiplicaciones entre columnas
    columnas = df.columns
    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):
            col1 = columnas[i]
            col2 = columnas[j]
            df_regresion[f"{col1}*{col2}"] = df[col1] * df[col2]

    # Calcular las sumatorias
    sumatorias = df_regresion.sum()
    df_regresion.loc["-------------"] = ["-" * 10] * df_regresion.shape[1]
    df_regresion.loc["Σ"] = sumatorias

    # Mostrar las sumatorias de manera organizada
    print("\n🔹 **Sumatorias de la tabla de regresión múltiple:**")
    for key, value in sumatorias.items():
        print(f"{key}: {value:.4f}")

    return df_regresion, sumatorias


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

def determinar_regresion(df_regresion, sumatorias):
    try:
        # Imprimir todas las claves de sumatorias para depuración
        print("\nClaves en sumatorias:")
        for key in sumatorias.keys():
            print(f"Clave en sumatorias: {key}")

        # Seleccionar las variables independientes
        columnas_independientes = [col for col in df_regresion.columns if "^2" not in col and "*" not in col and col != "y"]
        n = len(columnas_independientes)  # Número de variables independientes

        # Número de observaciones (filas), excluyendo la fila de sumatorias
        num_observaciones = len(df_regresion) - 2  # Excluir la fila de sumatorias y los titulos

        # Construcción de la matriz A 
        A = np.zeros((n, n)) 
        B = np.zeros(n)       

        # Primera fila de A
        A[0, 0] = num_observaciones
        for j in range(1, n):
            A[0, j] = sumatorias[columnas_independientes[j]]
            A[j, 0] = sumatorias[columnas_independientes[j]]  # Primera columna es igual a la primera fila

        # Resto de la matriz A
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    # Diagonal principal: sumatorias de los cuadrados
                    A[i, j] = sumatorias[f"{columnas_independientes[i]}^2"]
                else:
                    # Fuera de la diagonal: sumatorias de los productos cruzados
                    clave = f"{min(columnas_independientes[i], columnas_independientes[j])}*{max(columnas_independientes[i], columnas_independientes[j])}"
                    A[i, j] = sumatorias[clave]

        # Construir el vector B con las sumatorias de las variables independientes
        B[0] = sumatorias[columnas_independientes[0]]  # Σx1
        for i in range(1, n):
            clave = f"{columnas_independientes[0]}*{columnas_independientes[i]}"
            B[i] = sumatorias[clave] 

        # Mostrar la matriz ampliada [A|B]
        matriz_ampliada = np.hstack([A, B.reshape(-1, 1)])  # Concatenar A y B
        print("\nMatriz ampliada [A|B]:")
        print(tabulate(matriz_ampliada, headers=[f"B{i}" for i in range(n)] + ["B"], tablefmt='grid', floatfmt='.4f'))


        # Determinante para verificar si A es invertible
        det_A = np.linalg.det(A)
        print(f"\nDeterminante de A: {det_A}")
        if np.isclose(det_A, 0):
            raise ValueError("La matriz A no es invertible, el sistema no tiene solución única.")

        # Ejecutar Gauss-Jordan
        matriz_final = gauss_jordan(A, B)
        
        # Mostrar resultados finales
        print("\nMatriz final:")
        print(tabulate(matriz_final, headers=[f"B{i}" for i in range(n)] + ["B"], tablefmt='grid', floatfmt='.4f'))

        resultados = matriz_final[:, -1]
        print("\nResultados:")
        for i in range(n):
            print(f"β{i} = {resultados[i]:.4f}")
        
        # Construcción de la ecuación de regresión
        ecuacion = "y = " + " + ".join([f"{resultados[i]:.4f}*{columnas_independientes[i]}" for i in range(n)])
        print("\nRecta de regresión múltiple:")
        print(ecuacion)

        return resultados
    
    except Exception as e:
        import traceback
        print(f"Error al calcular la regresión: {e}")
        print(traceback.format_exc())
        return None
    
    
df_regresion, sumatorias = crear_tabla_regresion(df)
print("\nTabla de Contingencia con Datos Calculados:")
print(df_regresion)
coeficientes = determinar_regresion(df_regresion, sumatorias)

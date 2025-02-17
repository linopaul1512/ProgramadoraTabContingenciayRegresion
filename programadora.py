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
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools


fig, ax = plt.subplots(1, 1)


"""Listas para almacenar los datos ingresados"""
oxido_nitrosoy = []
humedadx1 = []
temperaturax2 = []
presionx3 = []


"""Funcion para pedir y tratar las listas"""
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


"""Consola de usuario"""
y = ingresar_listas("y (Óxido nitroso)")
x1 = ingresar_listas("x1 (Humedad)")
x2 = ingresar_listas("x2 (Temperatura)")
x3 = ingresar_listas("x3 (Presión)")


"""Agregar los valores a las listas"""
oxido_nitrosoy.extend(y)
humedadx1.extend(x1)
temperaturax2.extend(x2)
presionx3.extend(x3)


"""Determinar automáticamente el número de columnas (t)"""
columnas = [oxido_nitrosoy, humedadx1, temperaturax2, presionx3]
t = len(columnas)

"""Solicitar el nivel de significancia después del ingreso de datos"""
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


""" Verificar que todas las listas tengan la misma cantidad de elementos """
min_length = min(len(y), len(x1), len(x2), len(x3))

""" Ajustar las listas para que tengan el mismo tamaño """
y = y[:min_length]
x1 = x1[:min_length]
x2 = x2[:min_length]
x3 = x3[:min_length]

""" Crear DataFrame con pandas """
df = pd.DataFrame({
    "Óxido Nitroso (y)": y,
    "Humedad (x1)": x1,
    "Temperatura (x2)": x2,
    "Presión (x3)": x3
})

print(df)

"""Σxt sumatorias de los elementos de cada columna"""
Σxt_oxido_nitrosoy = sum(oxido_nitrosoy)
Σxt_humedadx1  = sum(humedadx1)
Σxt_temperaturax2  = sum(temperaturax2)
Σxt_presionx3  = sum(presionx3)

print(f"Σyt: {round(Σxt_oxido_nitrosoy, 4)}, Σx1t humedadx1: {round(Σxt_humedadx1, 4)} Σx2t humedadx1: {round(Σxt_temperaturax2, 4)}, Σx3t: {round(Σxt_presionx3, 4)} ")

Σxt_4col = sum([Σxt_oxido_nitrosoy,Σxt_humedadx1, Σxt_temperaturax2, Σxt_presionx3])

print ("Σxt", round(Σxt_4col, 4))


"""Sumatorias de los elementos elevados al cuadrado"""
Σxt2_oxido_nitrosoy = sum([elemento ** 2 for elemento in oxido_nitrosoy])
Σxt2_humedadx1 = sum([elemento ** 2 for elemento in humedadx1])
Σxt2_temperaturax2 = sum([elemento ** 2 for elemento in temperaturax2])
Σxt2_presionx3 = sum([elemento ** 2 for elemento in presionx3])

print(f" Σyt²: {round(Σxt2_oxido_nitrosoy, 4)}, Σx1t humedadx1: {round(Σxt2_humedadx1, 4)}, Σx2t² temperaturax2: {round(Σxt2_temperaturax2, 4)}, Σx3t² presionx3: {round(Σxt2_presionx3, 4)}")


Σxt2_4col = sum([Σxt2_oxido_nitrosoy,Σxt2_humedadx1, Σxt2_temperaturax2, Σxt2_presionx3])

print ("Σxt²: ", round(Σxt2_4col, 4))
  

"""Sumatorias de las sumas de los lementos de las muestras elevado al cuadrado"""
Σxtcua_oxido_nitrosoy =  sum(oxido_nitrosoy)  ** 2
Σxtcua_humedadx1  = sum(humedadx1) ** 2
Σxtcua_temperaturax2  =  sum(temperaturax2) ** 2
Σxtcua_presionx3  = sum(presionx3) ** 2

print(f" (Σyt)²: {round(Σxtcua_oxido_nitrosoy, 4)}, Σx1t humedadx1: {round(Σxtcua_humedadx1, 4)}, Σx2t² temperaturax2: {round(Σxtcua_temperaturax2, 4)}, Σx3t² presionx3: {round(Σxtcua_presionx3, 4)}")

Σxtcua_4col = sum([Σxtcua_oxido_nitrosoy,Σxtcua_humedadx1, Σxtcua_temperaturax2, Σxtcua_presionx3])

print ("(Σxt)²: ", round(Σxtcua_4col, 4))

"""Sumatoria de la cantidad de elementos por columna"""
nt_oxido_nitrosoy = len(oxido_nitrosoy)
nt_humedadx1 =  len(humedadx1)
nt_temperaturax2 = len(temperaturax2)
nt_presionx3 = len(presionx3)

print(f" nty: {nt_oxido_nitrosoy}, ntx1: {round(nt_humedadx1)}, ntx2: {nt_temperaturax2}, ntx3: {nt_presionx3}")

Σnt_4col = sum([nt_oxido_nitrosoy,nt_humedadx1, nt_temperaturax2, nt_presionx3])

print ("Σnt: ", Σnt_4col)

"""Sumatorias de los elementos elevados al cuadrado divido entre n (cantidad de elementos)"""
ΣxtcuaN_oxido_nitrosoy =  Σxtcua_oxido_nitrosoy /nt_oxido_nitrosoy
ΣxtcuaN_humedadx1  = Σxtcua_humedadx1 / nt_humedadx1
ΣxtcuaN_temperaturax2  = Σxtcua_temperaturax2 / nt_temperaturax2
ΣxtcuaN_presionx3  = Σxtcua_presionx3 / nt_presionx3

print(f" (Σyt)²/n: {round(ΣxtcuaN_oxido_nitrosoy, 4)}, (Σx1t)²/n: {round(ΣxtcuaN_humedadx1, 4)}, (Σx2t)²/n: { round(ΣxtcuaN_temperaturax2, 4)}, (Σx3t)²/n: {round(ΣxtcuaN_presionx3, 4)}")


ΣxtcuaN_4col = sum([ΣxtcuaN_oxido_nitrosoy, ΣxtcuaN_humedadx1, ΣxtcuaN_temperaturax2, ΣxtcuaN_presionx3])
print (" (Σxt)²/n: ", ΣxtcuaN_4col)

"""Sumatoria de las media aritmeticas"""
x_oxido_nitrosoy = Σxt_oxido_nitrosoy / nt_oxido_nitrosoy
x_humedadx1 = Σxt_humedadx1 / nt_humedadx1
x_temperaturax2 = Σxt_temperaturax2 / nt_temperaturax2
x_presionx3 =  Σxt_presionx3 / nt_presionx3

print(f" x̅y: {round(x_oxido_nitrosoy, 4)}, x̅x1: {round(x_humedadx1, 4)}, x̅x2: {round(x_temperaturax2)}, x̅x3: {round(x_presionx3)}")

x_4col = sum([x_oxido_nitrosoy, x_humedadx1, x_temperaturax2, x_presionx3])
print ("x̅: ", round(x_4col))

"""Calcular nivel de significancia"""
alfa = 1 - nivel_significancia
print(f"α (alfa):" , round(alfa, 4))

"""grados de libertad del tratamiento"""
gl_tratamiento =   t -1
print(f"gl(tratamiento):" , gl_tratamiento)

"""grados de libertad del error"""
gl_error = Σnt_4col - t
print(f"gl(error):" , round(gl_error, 4))

"""Factor de correcion"""
c =  Σxt_4col ** 2  / Σnt_4col
print(f"Factor de correcion (C):" , round(c, 4))

"""Suma Total de Cuadradados"""
sct = Σxt2_4col - c
print(f"Suma Total de Cuadrados (SCT):" , round(sct, 4))

"""Suma Cuadradada de Tratamiento"""
sctr = ΣxtcuaN_4col - c
print(f"Suma Cuadradada de Tratamiento (SCTR):" , round(sctr, 4))

"""Suma Cuadradada de error"""
sce = Σxt2_4col - ΣxtcuaN_4col
print(f"Suma Cuadradada de Error (SCE):" , round(sce, 4))

"""n - 1"""
nmenos1 = Σnt_4col - 1

"""MCTR"""
mctr = sctr / gl_tratamiento

"""MCE"""
mce = sce / gl_error

"""F(RV) Fisher razón de variacion"""
f_rv = mctr / mce


""" Crear DataFrame con pandas de la fuente de variacion"""

fuente_variacion = pd.DataFrame({
    "Fuentes de variacion": ["Tratamiento", "Error", "Total"],
    "SC": [round(sctr, 4), round(sce, 4), round(sct, 4)],
    "gl": [round(gl_tratamiento, 4), round(gl_error, 4), nmenos1],
    "MC": [round(mctr, 4), round(mce, 4), None],  
    "F(RV)": [round(f_rv, 4), None, None]  
})


print(fuente_variacion)



"""Buscar F tabulada """
Ftab = stats.f.ppf(1 - alfa, gl_tratamiento, gl_error)
print(f"F tabulada: {round(Ftab, 4)}")

"""Comparación y decision"""
print("[Si  Fcal > Ftab = RR]" , "Fcalc < Ftab = RA")
if f_rv > Ftab:
    decision = "Rechazar H₀ (Existe diferencia significativa)"
else:
    decision = "Aceptar H₀ (No hay diferencia significativa)"

print(f"Decisión: {decision}")

"""Prueba DHS"""
# Número de grupos (columnas)
num_grupos = t

# Valor crítico q para HSD
q = studentized_range.ppf( 1 - alfa, num_grupos, gl_error)

# Calcular DHS
dhs = q * np.sqrt(mce / nt_oxido_nitrosoy)  # Usamos uno de los n's (asumido igual para todos)
print()
print(f"Valor crítico q: {q}")
print(f"Diferencia Honestamente Significativa (HSD): {dhs}")



"""Tukey """

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


# Crear la tabla 
print("Comparación de Medias - Prueba de Tukey\n")
print(f"{'Grupo 1':<15}{'Grupo 2':<15}{'Diferencia':<15}{'DHS':<10}{'Independencia'}")
print("-" * 65)

for g1, g2 in pares:
    meandiff = medias[g1] - medias[g2]  # Diferencia de medias
    independencia = "Independiente" if meandiff > dhs or meandiff > -dhs else "Dependiente"
    
    # Mostrar los resultados en formato de tabla
    print(f"{g1:<15}{g2:<15}{meandiff:<15.4f}{dhs:<10.4f}{independencia}")


#Correlacion  y recta de regresion lineal

 




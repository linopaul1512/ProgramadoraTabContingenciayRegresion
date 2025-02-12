import numpy as np
import pandas as pd


# Listas para almacenar los datos ingresados
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
                """Convertimos la entrada en una lista de números"""
                lista_numeros = [float(valor) for valor in valores.split(",")]
                return lista_numeros
            except ValueError:
                print("⚠️ Error: Debe ingresar solo números separados por comas.")
        else:
            print("⚠️ Este campo es obligatorio. Intente de nuevo.")

print("\n🔹 Ingrese los valores fila por fila (deje vacío para finalizar)")

"""Consola de usuario"""
while True:
    print("\n Nueva fila:")
    y = ingresar_listas("y (Óxido nitroso)")
    x1 = ingresar_listas("x1 (Humedad)")
    x2 = ingresar_listas("x2 (Temperatura)")
    x3 = ingresar_listas("x3 (Presión)")

    """Agregar los valores a las listas"""
    oxido_nitrosoy.extend(y)
    humedadx1.extend(x1)
    temperaturax2.extend(x2)
    presionx3.extend(x3)


    continuar = input("¿Desea agregar otra fila? (s/n): ").strip().lower()
    if continuar != 's':
        break


"""Crear tabla con los datos"""



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

print(f" nty: {nt_oxido_nitrosoy}, ntx1: {round(nt_humedadx1, 4)}, ntx2: {nt_temperaturax2}, ntx3: {nt_presionx3}")

Σnt_4col = sum([nt_oxido_nitrosoy,nt_humedadx1, nt_temperaturax2, nt_presionx3])

print ("Σnt: ", Σnt_4col)


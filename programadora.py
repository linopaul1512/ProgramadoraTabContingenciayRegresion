import numpy as np
import numpy as np

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


"""Σxt sumatorias de los elementos de cada columna"""

Σxt_oxido_nitrosoy = sum(oxido_nitrosoy)
Σxt_humedadx1  = sum(humedadx1)
Σxt_temperaturax2  = sum(temperaturax2)
Σxt_presionx3  = sum(presionx3)

print("Σyt", Σxt_oxido_nitrosoy, "Σx1t humedadx1", Σxt_humedadx1, "Σx2t humedadx1", Σxt_temperaturax2, "Σx3t", Σxt_presionx3 )

Σxt_4col = sum([Σxt_oxido_nitrosoy,Σxt_humedadx1, Σxt_temperaturax2, Σxt_presionx3])

print ("Σxt", Σxt_4col)


"""Sumatorias de los elementos elevados al cuadrado"""
Σxt2_oxido_nitrosoy = sum([elemento ** 2 for elemento in oxido_nitrosoy])
Σxt2_humedadx1 = sum([elemento ** 2 for elemento in humedadx1])
Σxt2_temperaturax2 = sum([elemento ** 2 for elemento in temperaturax2])
Σxt2_presionx3 = sum([elemento ** 2 for elemento in presionx3])

print("Σyt²:", Σxt2_oxido_nitrosoy, 
      "Σx1t² humedadx1:", Σxt2_humedadx1, 
      "Σx2t² temperaturax2:", Σxt2_temperaturax2, 
      "Σx3t² presionx3:", Σxt2_presionx3)
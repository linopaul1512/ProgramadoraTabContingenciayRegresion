import pandas as pd
import numpy as np

oxido_nitrosoy =  [0.90, 0.91, 0.96, 0.89, 1.00, 1.10, 1.15, 1.03, 0.77, 1.07, 1.07, 0.94, 1.10, 1.10, 1.10, 0.91, 0.87, 0.78, 0.82, 0.95, 1.05, 1.01, 0.96, 0.87, 0.96, 0.88, 1.10, 0.98, 0.89, 1.06]
humedadx1 =[72.4, 41.6, 34.3, 35.1, 10.7, 12.9, 8.3, 20.1, 72.2, 24, 23.2, 47.4, 31.5, 10.6, 11.2, 73.3, 75.4, 96.6, 107.4, 54.9, 70, 56, 39, 36.4, 41, 45, 11.2, 41, 41.3, 42.32]
temperaturax2 = [76.3, 70.3, 77.1, 68.0, 79.0, 79.0, 66.8, 76.9, 77.7, 67.7, 76.8, 86.6, 76.9, 86.3, 86.0, 76.3, 77.9, 78.7,  86.8, 70.9, 72.2, 73.1, 75.1, 74.2, 74.9, 74.0, 87.2, 74.8, 76.2, 77.1]
presionx3=[29.18, 29.35, 29.24, 29.27, 29.78, 29.39, 29.69, 29.48, 29.09, 29.60, 29.38, 29.35, 29.63, 29.56, 29.48, 29.40, 29.28, 29.29, 29.03, 29.37, 28.10, 29.20, 29.50, 29.12, 29.21, 28.99, 29.86, 29.02, 29.51, 29.32]


"""Σxt sumatorias de los elementos de cada columna"""


Σxt_oxido_nitrosoy = 0 
Σxt_humedadx1 = 0
Σxt_temperaturax2 = 0
Σxt_presionx3 = 0

for  Σxt  in oxido_nitrosoy, humedadx1, temperaturax2, presionx3:
    Σxt_oxido_nitrosoy = sum(oxido_nitrosoy)
    Σxt_humedadx1  = sum(humedadx1)
    Σxt_temperaturax2  = sum(temperaturax2)
    Σxt_presionx3  = sum(presionx3)

print("Σyt", Σxt_oxido_nitrosoy, "Σx1t humedadx1", Σxt_humedadx1, "Σx2t humedadx1", Σxt_temperaturax2, "Σx3t", Σxt_presionx3 )

Σxt_4col = sum([Σxt_oxido_nitrosoy,Σxt_humedadx1, Σxt_temperaturax2, Σxt_presionx3])

print ("Σxt", Σxt_4col)

"""Σxt² sumatorias de los elementos elevados al cuadrado"""

Σxt2_oxido_nitrosoy = list( map( lambda elemento : elemento * elemento , [oxido_nitrosoy]) )
Σxt2_humedadx1 = list( map( lambda elemento : elemento * elemento , [Σxt_humedadx1] ) )
Σxt2_temperaturax2 = list( map( lambda elemento : elemento * elemento , [Σxt_temperaturax2]) )
Σxt2_presionx3 = list( map( lambda elemento : elemento * elemento , [Σxt_presionx3] ) )

print("Σyt²", Σxt2_oxido_nitrosoy, "Σx1t² humedadx1", Σxt2_humedadx1, "Σx2t² humedadx1", Σxt2_temperaturax2, "Σx3t²", Σxt2_presionx3 )



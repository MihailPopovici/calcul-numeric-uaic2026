"""
Calculatoarele reprezinta numerele in virgula mobila.
Asta inseamna ca au o precizie FINITA. Daca adunam un numar suficient de mic
la 1.0, calculatorul nu "vede" diferenta si rezultatul ramane 1.0.

Cautam cel mai mic u = 10^(-m), cu m natural, astfel incat:
    1.0 + u != 1.0   (in aritmetica calculatorului)

Strategia: crestem m (scadem u) pana cand 1.0 + u == 1.0, 
           apoi ne oprim la valoarea anterioara.
"""

def machine_precision():
    m = 0
    while True:
        u = 10 ** (-m)
        if 1.0 + u == 1.0:
            return 10 ** (-(m - 1))
        m += 1

u = machine_precision()
print("Precizia masina:", u) #1e-15 = 1*10^(-15)
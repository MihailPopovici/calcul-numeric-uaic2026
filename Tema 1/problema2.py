print("2a) Adunarea nu este asociativa")
"""
Se iau x = 1.0, y = u/10, z = u/10
unde u este precizia masinii (pentru care 1.0 + u/10 == 1.0).

(x +c y) +c z = (1.0 + u/10) + u/10
              = 1.0 + u/10          (deoarece 1.0 + u/10 == 1.0)
              = 1.0                  (din nou, u/10 prea mic)

x +c (y +c z) = 1.0 + (u/10 + u/10)
              = 1.0 + u/5
              = 1.0 + 2*u/10
              Dar 2*u/10 = u/5 > u/10, si daca u/5 este suficient
              de mare, 1.0 + u/5 != 1.0  → rezultate diferite!
"""
def machine_precision():
    m = 0
    while True:
        u = 10 ** (-m)
        if 1.0 + u == 1.0:
            return 10 ** (-(m - 1))
        m += 1

u = machine_precision()

x = 1.0
y = u / 10
z = u / 10

stanga = (x + y) + z    # (x +c y) +c z
dreapta = x + (y + z)   # x +c (y +c z)

print(f"  x = {x}, y = u/10 = {y}, z = u/10 = {z}")
print(f"  (x + y) + z = {stanga}")
print(f"  x + (y + z) = {dreapta}")
print(f"  Sunt egale? {stanga == dreapta}")
print(f"  Neasociativa? {stanga != dreapta}\n")

print("="*60)

print("\n--- 2b) Inmultirea nu este asociativa ---")

"""

Idea: Luam numere mari si mici astfel incat ordinea operatiilor
sa produca overflow sau underflow.

Exemplu:
  x = 1e200, y = 1e200, z = 1e-200

  (x * y) * z = (1e200 * 1e200) * 1e-200
             = 1e400 * 1e-200   → 1e400 = inf (overflow)
             = inf * 1e-200     = inf

  x * (y * z) = 1e200 * (1e200 * 1e-200)
             = 1e200 * 1e0
             = 1e200             (rezultat finit, corect)
"""

x2 = 1e200
y2 = 1e200
z2 = 1e-200

stanga2 = (x2 * y2) * z2
dreapta2 = x2 * (y2 * z2)

print(f"  x = {x2}, y = {y2}, z = {z2}")
print(f"  (x * y) * z = {stanga2}  (overflow -> inf)")
print(f"  x * (y * z) = {dreapta2}  (rezultat corect)")
print(f"  Neasociativa? {stanga2 != dreapta2}")

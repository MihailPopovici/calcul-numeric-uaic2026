"""
Tema nr. 2 - Descompunerea Cholesky (LDLT)
==========================================

Rezolvarea sistemului Ax = b folosind descompunerea LDLT,
unde A este matrice simetrica si pozitiv definita.

Structura programului:
1. Generare matrice A = B * B^T (simetrica, pozitiv definita)
2. Rezolvare cu biblioteca (LU) -> xlib
3. Calcul descompunere LDLT (Cholesky)
4. Calcul determinant folosind LDLT
5. Rezolvare sistem Ax = b folosind LDLT -> xChol
6. Verificare solutie prin norme
"""

import numpy as np
import math

# ============================================================
# 1. CITIRE DATE DE INTRARE
# ============================================================

print("=" * 60)
print("  TEMA 2 - Descompunerea Cholesky (LDLT)")
print("=" * 60)

n   = int(input("\nIntroduceti dimensiunea sistemului n: "))
t   = int(input("Introduceti precizia t (eps = 10^(-t), ex: t=8): "))
eps = 10 ** (-t)
print(f"\nn = {n},  eps = 1e-{t} = {eps}")

# ============================================================
# 2. GENERARE MATRICE A SIMETRICA SI POZITIV DEFINITA
#    A = B * B^T
# ============================================================

np.random.seed(42)                        # reproductibilitate
B      = np.random.uniform(-10, 10, (n, n))
A_init = B @ B.T                          # A simetrica si p.d.

# Adaugam n*I pentru a ne asigura ca A este bine conditionata
# (optional, dar recomandat pentru n mare)
A_init += n * np.eye(n)

"""
B @ B.T e pozitiv semi-definit, adica exista riscul teoretic ca unele valori din d sa fie 0 sau extrem de mici,
ceea ce ar strica impartirea din algoritmul Cholesky. 
Adaugand n*I "impingem" toata matricea departe de această zona periculoasă. Matricea ramane simetrica si devine strict pozitiv definita.
"""

b = np.random.uniform(-10, 10, n)         # vector termeni liberi

print(f"\nMatricea A a fost generata ca B*B^T + {n}*I (simetrica, pozitiv definita).")

# ============================================================
# 3. REZOLVARE CU BIBLIOTECA NUMPY (LU implicit prin np.linalg)
# ============================================================

print("\n--- Rezolvare cu biblioteca numpy (np.linalg.solve) ---")
xlib = np.linalg.solve(A_init, b)
k = min(5, n)
print(f"xlib (primele {k} valori): {xlib[:5]}")

# ============================================================
# 4. DESCOMPUNEREA LDLT (CHOLESKY)
#    Lucram pe o copie a lui A_init.
#    Elementele lui L (sub-diagonala) se scriu direct in A.
#    Diagonala lui D se memoreaza in vectorul d.
#    Diagonala lui L (1-uri) NU se memoreaza explicit.
#    Partea superior triunghiulara a lui A ramane neschimbata
#    (contine elementele originale ale lui A_init).
# ============================================================

print("\n--- Calcul descompunere LDLT ---")

A = A_init.copy()          # matricea de lucru (va fi modificata)
d = np.zeros(n)            # vectorul diagonalei lui D

success = True

for p in range(n):         # p = 0, 1, ..., n-1  (indexare 0-based)

    # --- Calculul elementului diagonal dp ---
    # dp = a[p][p] - sum_{k=0}^{p-1} d[k] * l[p][k]^2
    # l[p][k] se afla in A[p][k] pentru k < p (deja calculat)
    s = A[p, p]
    for k in range(p):
        s -= d[k] * A[p, k] ** 2

    # Test daca dp este suficient de diferit de 0
    if math.fabs(s) <= eps:
        print(f"EROARE: d[{p}] = {s} <= eps. Matricea nu este pozitiv definita!")
        success = False
        break

    d[p] = s

    # --- Calculul elementelor coloanei p din L: l[i][p], i = p+1, ..., n-1 ---
    # l[i][p] = (a[i][p] - sum_{k=0}^{p-1} d[k]*l[i][k]*l[p][k]) / dp
    for i in range(p + 1, n):
        t_val = A[i, p]                   # a[i][p] original (sus-diag: A[p][i] = A_init[p][i])
        # Atentie: A[i][p] pt i>p se va suprascrie -> folosim A[p][i] pt elementul original
        # De fapt: pentru i > p, A[i][p] inca nu a fost suprascris la pasul curent
        # dar ca sa fim siguri, citim elementul original din partea superior-triunghiulara
        t_val = A[p, i]                   # A este simetrica initial, A[p][i] = A_init[p][i]
        for k in range(p):
            t_val -= d[k] * A[i, k] * A[p, k]
        # A[p][k] pentru k<p = l[p][k], deja calculat si stocat in A

        if math.fabs(d[p]) > eps:
            A[i, p] = t_val / d[p]
        else:
            print(f"EROARE la impartire: d[{p}] prea mic!")
            success = False
            break

    if not success:
        break

if success:
    print("Descompunerea LDLT calculata cu succes!")
    k = min(5, n)
    print(f"d (primele {k} valori): {d[:5]}")

# ============================================================
# 5. CALCULUL DETERMINANTULUI
#    det(A) = det(L) * det(D) * det(L^T)
#    det(L) = 1  (L inferior triunghiulara cu 1 pe diagonala)
#    det(L^T) = 1
#    det(D) = d[0] * d[1] * ... * d[n-1]
#    => det(A) = prod(d)
# ============================================================

print("\n--- Calculul determinantului ---")

det_A_chol = math.prod(d)                  # det(A) = det(L)*det(D)*det(L^T) = 1*prod(d)*1
det_A_lib  = np.linalg.det(A_init)

print(f"det(A) prin LDLT    : {det_A_chol:.6e}")
print(f"det(A) numpy        : {det_A_lib:.6e}")
print(f"Eroare relativa     : {abs(det_A_chol - det_A_lib) / (abs(det_A_lib) + 1e-300):.2e}")

# ============================================================
# 6. REZOLVAREA SISTEMULUI Ax = b FOLOSIND LDLT
#
#    Ax = b  <=>  L D L^T x = b
#    Se rezolva in 3 pasi:
#      (a) Lz = b      (substitutie directa, diagonala L = 1)
#      (b) Dy = z      (sistem diagonal: y[i] = z[i] / d[i])
#      (c) L^T x = y   (substitutie inversa, diagonala L^T = 1)
#
#    In matricea A modificata:
#      A[i][j] pentru i > j  => l[i][j]  (elementele lui L)
#      A[i][j] pentru i <= j => a_init[i][j] (elementele originale)
# ============================================================

print("\n--- Rezolvare sistem Ax = b cu LDLT ---")

if success:

    # --- (a) Substitutie directa: Lz = b ---
    # L inferior triunghiulara cu l[i][i] = 1
    # z[i] = b[i] - sum_{j=0}^{i-1} l[i][j] * z[j]
    # l[i][j] se afla in A[i][j] pentru j < i
    z = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= A[i, j] * z[j]      # A[i][j] = l[i][j] pentru j < i
        z[i] = s                     # diagonala 1, deci nu impartim

    # --- (b) Sistem diagonal: Dy = z ---
    # y[i] = z[i] / d[i]
    y = np.zeros(n)
    for i in range(n):
        if math.fabs(d[i]) > eps:
            y[i] = z[i] / d[i]
        else:
            print(f"EROARE: d[{i}] = 0 la rezolvarea Dy = z")
            success = False
            break

    # --- (c) Substitutie inversa: L^T x = y ---
    # L^T superior triunghiulara cu l^T[i][i] = 1
    # x[i] = y[i] - sum_{j=i+1}^{n-1} l^T[i][j] * x[j]
    #       = y[i] - sum_{j=i+1}^{n-1} l[j][i] * x[j]
    # l[j][i] se afla in A[j][i] pentru j > i
    xChol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= A[j, i] * xChol[j]  # A[j][i] = l[j][i] pentru j > i
        xChol[i] = s                 # diagonala 1, deci nu impartim
    k = min(5, n)
    print(f"xChol (primele {k} valori): {xChol[:5]}")

# ============================================================
# 7. VERIFICARE - CALCULUL NORMELOR
#    ||A_init * xChol - b||_2  (ar trebui < 1e-8)
#    ||xChol - xlib||_2        (ar trebui < 1e-9)
#
#    IMPORTANT: A_init * xChol trebuie calculat manual,
#    deoarece matricea A a fost modificata de Cholesky.
#    Se foloseste simetria: a[i][j] = a[j][i] pentru i > j.
#    In matricea A modificata:
#      - A[i][j] cu i > j contine l[i][j] (NU a_init[i][j])
#      - A[i][j] cu i <= j contine a_init[i][j]
#    Deci pentru i > j folosim A[j][i] (elementul din partea superioara).
# ============================================================

print("\n--- Verificare solutie ---")

if success:

    # Inmultire matrice-vector: y = A_init * xChol
    # Se foloseste A_init explicit (nu A modificata)
    y_check = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i <= j:
                s += A[i, j] * xChol[j]      # A[i][j] = a_init[i][j] (zona superioara)
            else:
                s += A[j, i] * xChol[j]      # A[j][i] = a_init[i][j] (simetrie)
        y_check[i] = s

    # Norma 1: ||A_init * xChol - b||_2
    diff1 = y_check - b
    norm1 = math.sqrt(sum(x * x for x in diff1))

    # Norma 2: ||xChol - xlib||_2
    diff2 = xChol - xlib
    norm2 = math.sqrt(sum(x * x for x in diff2))

    print(f"||A_init * xChol - b||_2 = {norm1:.6e}   (target: < 1e-8)")
    print(f"||xChol - xlib||_2       = {norm2:.6e}   (target: < 1e-9)")

    if norm1 < 1e-8:
        print("Prima norma este in limita acceptabila (< 1e-8)")
    else:
        print("Prima norma DEPASESTE limita de 1e-8!")

    if norm2 < 1e-9:
        print("A doua norma este in limita acceptabila (< 1e-9)")
    else:
        print("A doua norma DEPASESTE limita de 1e-9!")

# ============================================================
# 8. TEST PE EXEMPLUL DIN PDF
#    A = [[1, 2.5, 3], [2.5, 8.25, 15.5], [3, 15.5, 43]]
#    L = [[1,0,0],[2.5,1,0],[3,4,1]], D = diag(1,2,2)
# ============================================================

print("\n" + "=" * 60)
print("  TEST PE EXEMPLUL DIN PDF (n=3)")
print("=" * 60)

A_ex = np.array([[1.0, 2.5, 3.0],
                 [2.5, 8.25, 15.5],
                 [3.0, 15.5, 43.0]])
b_ex = np.array([1.0, 2.0, 3.0])
n_ex = 3
eps_ex = 1e-8

A_ex_work = A_ex.copy()
d_ex = np.zeros(n_ex)

for p in range(n_ex):
    s = A_ex_work[p, p]
    for k in range(p):
        s -= d_ex[k] * A_ex_work[p, k] ** 2
    d_ex[p] = s
    for i in range(p + 1, n_ex):
        t_val = A_ex_work[p, i]
        for k in range(p):
            t_val -= d_ex[k] * A_ex_work[i, k] * A_ex_work[p, k]
        A_ex_work[i, p] = t_val / d_ex[p]

print(f"\nD diagonal (asteptat [1, 2, 2]): {d_ex}")
print(f"L sub-diagonala col 0 (asteptat [2.5, 3.0]): {A_ex_work[1,0]:.4f}, {A_ex_work[2,0]:.4f}")
print(f"L sub-diagonala col 1 (asteptat [4.0]):       {A_ex_work[2,1]:.4f}")

# Rezolvare sistem exemplu
z_ex = np.zeros(n_ex)
for i in range(n_ex):
    s = b_ex[i]
    for j in range(i):
        s -= A_ex_work[i, j] * z_ex[j]
    z_ex[i] = s

y_ex = np.zeros(n_ex)
for i in range(n_ex):
    y_ex[i] = z_ex[i] / d_ex[i]

x_ex = np.zeros(n_ex)
for i in range(n_ex - 1, -1, -1):
    s = y_ex[i]
    for j in range(i + 1, n_ex):
        s -= A_ex_work[j, i] * x_ex[j]
    x_ex[i] = s

xlib_ex = np.linalg.solve(A_ex, b_ex)
print(f"\nxChol (exemplu) : {x_ex}")
print(f"xlib  (exemplu) : {xlib_ex}")
print(f"Diferenta norma : {np.linalg.norm(x_ex - xlib_ex):.2e}")

print("\n" + "=" * 60)
print("  PROGRAM TERMINAT")
print("=" * 60)
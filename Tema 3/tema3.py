"""
Tema nr. 3 - Descompunerea QR folosind algoritmul Householder
=============================================================

Structura programului:
1. Generare date: matrice A aleatoare, vector s, calcul b = A*s
2. Descompunere QR cu biblioteca numpy -> xQR
3. Descompunere QR cu algoritmul Householder implementat manual -> xHouseholder
4. Calcul erori si verificare
5. Calcul inversa folosind Householder
"""

import numpy as np
import math

print("=" * 60)
print("  TEMA 3 - Descompunerea QR (Algoritmul Householder)")
print("=" * 60)

# ============================================================
# CITIRE DATE DE INTRARE
# ============================================================

n   = int(input("\nIntroduceti dimensiunea sistemului n: "))
t   = int(input("Introduceti precizia t (eps = 10^(-t), ex: t=8): "))
eps = 10 ** (-t)
print(f"\nn = {n},  eps = 1e-{t} = {eps}")

# ============================================================
# 1. GENERARE DATE
#    - A = matrice patratica aleatoare n x n
#    - s = vectorul solutie exacta (il stim dinainte!)
#    - b = A * s  (calculat ca suma: b[i] = sum_j s[j]*a[i][j])
#
#    Trucul: daca construim b = A*s, atunci solutia exacta
#    a sistemului Ax = b este chiar s. Asta ne permite sa
#    verificam cat de precisa e solutia noastra la final.
# ============================================================

np.random.seed(42)
A_init = np.random.uniform(-10, 10, (n, n))
s      = np.random.uniform(-10, 10, n)

# Calculam b[i] = sum_{j=1}^{n} s[j] * a[i][j]
# (formula din tema, indexare 1-based in PDF, 0-based in cod)
b_init = np.zeros(n)
for i in range(n):
    for j in range(n):
        b_init[i] += s[j] * A_init[i, j]

# Verificare: trebuie sa fie acelasi lucru ca A_init @ s
# (folosim versiunea vectorizata doar pentru verificare interna)
assert np.allclose(b_init, A_init @ s), "Eroare la calculul lui b!"
print("\nVectorul b = A*s calculat cu succes.")

# ============================================================
# 2. REZOLVARE CU BIBLIOTECA NUMPY - Descompunere QR explicita
#    numpy.linalg.qr returneaza explicit Q si R
#    astfel incat A = Q * R
#    Rezolvam Ax = b <=> Q*R*x = b <=> R*x = Q^T * b
# ============================================================

print("\n--- Rezolvare cu biblioteca numpy (QR explicit) ---")

def substitutie_inversa(R, rhs, n, eps):
    """
    Rezolva sistemul superior triunghiular R*x = rhs.
    Returneaza vectorul solutie x sau None daca R e singulara.
    """
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if math.fabs(R[i, i]) <= eps:
            print(f"EROARE substitutie inversa: R[{i}][{i}] ~ 0!")
            return None
        s_val = rhs[i]
        for j in range(i + 1, n):
            s_val -= R[i, j] * x[j]
        x[i] = s_val / R[i, i]
    return x

def rezolvare_QR_numpy(A, b):
    """
    Rezolva sistemul Ax = b prin descompunere QR explicita.
    
    Pasi:
      1. A = Q * R  (Q ortogonala, R superior triunghiulara)
      2. Q*R*x = b  =>  R*x = Q^T * b   (deoarece Q^T * Q = I)
      3. Rezolvam R*x = Q^T * b prin substitutie inversa
    """
    n = len(b)

    # Pasul 1: Descompunere QR explicita
    # Q: matrice ortogonala (n x n), R: matrice superior triunghiulara (n x n)
    Q, R = np.linalg.qr(A)

    print(f"Q shape: {Q.shape}, R shape: {R.shape}")
    print(f"R[0][0] = {R[0,0]:.6f}  (element diagonal al lui R)")

    # Verificare: A ≈ Q * R
    eroare_reconstructie = np.linalg.norm(A - Q @ R)
    print(f"||A - Q*R||_2 = {eroare_reconstructie:.2e}  (verificare: A = Q*R)")

    # Verificare: Q este ortogonala (Q^T * Q = I)
    eroare_ortogonalitate = np.linalg.norm(Q.T @ Q - np.eye(n))
    print(f"||Q^T*Q - I||_2 = {eroare_ortogonalitate:.2e}  (verificare: Q ortogonala)")

    # Pasul 2: Calculam termenul drept transformat: c = Q^T * b
    c = Q.T @ b

    # Pasul 3: Rezolvam R*x = c prin substitutie inversa
    x = substitutie_inversa(R, c, n, eps)

    return x, Q, R

xQR, Q_lib, R_lib = rezolvare_QR_numpy(A_init, b_init)
k_display = min(5, n)
print(f"xQR (primele {k_display} valori): {xQR[:k_display]}")

# ============================================================
# 3. ALGORITMUL HOUSEHOLDER - implementare manuala
#
# Ideea: aplicam n-1 reflectii asupra lui A si b simultan.
# La fiecare pas r, "curatam" coloana r (punem zerouri sub diagonala).
# La final: A devine R (superior triunghiulara)
#           Q_bar devine Q^T
#           b devine Q^T * b_init
# ============================================================

print("\n--- Calcul descompunere QR cu algoritmul Householder ---")

# Lucram pe copii - nu modificam A_init si b_init
A = A_init.copy()
b = b_init.copy()

# Initializam Q_bar = I_n
# La final va contine Q^T (transpusa lui Q)
Q_bar = np.eye(n)

singular = False

# Parcurgem coloanele de la 0 la n-2 (n-1 pasi)
for r in range(n - 1):

    # ----------------------------------------------------------
    # PASUL r: construim vectorul u si constanta beta
    #
    # sigma = suma patratelor elementelor din coloana r,
    #         de la pozitia r in jos
    # ----------------------------------------------------------

    sigma = 0.0
    for i in range(r, n):
        sigma += A[i, r] ** 2

    # Daca sigma e aproape 0, coloana e deja "curata" -> sarim
    if math.fabs(sigma) <= eps:
        # Matricea e (aproape) singulara sau coloana e deja 0
        # In practica pentru matrice aleatoare nu se intampla
        singular = True
        print(f"ATENTIE: sigma = 0 la pasul r={r}. Matrice posibil singulara!")
        break

    # k = sqrt(sigma), cu semn opus lui a[r][r]
    # (alegem semnul opus pentru stabilitate numerica -
    #  evitam scaderea a doua numere aproape egale)
    k = math.sqrt(sigma)
    if A[r, r] > 0:
        k = -k

    # beta = sigma - k * a[r][r]
    # (beta = u^T * u / 2, folosit la normalizare)
    beta = sigma - k * A[r, r]

    # Vectorul u:
    # u[i] = 0         pentru i < r
    # u[r] = a[r][r] - k
    # u[i] = a[i][r]   pentru i > r
    u = np.zeros(n)
    u[r] = A[r, r] - k
    for i in range(r + 1, n):
        u[i] = A[i, r]

    # Verificam ca beta != 0 inainte de impartire
    if math.fabs(beta) <= eps:
        singular = True
        print(f"EROARE: beta = 0 la pasul r={r}!")
        break

    # ----------------------------------------------------------
    # TRANSFORMAREA COLOANELOR j = r+1, ..., n-1
    # (coloana r se seteaza direct mai jos)
    #
    # Pentru fiecare coloana j:
    #   gamma = (u^T * coloana_j) / beta = sum_i u[i]*a[i][j] / beta
    #   a[i][j] = a[i][j] - gamma * u[i]  pentru i = r..n-1
    # ----------------------------------------------------------

    for j in range(r + 1, n):
        # Calculam gamma = produsul scalar intre u si coloana j
        gamma = 0.0
        for i in range(r, n):
            gamma += u[i] * A[i, j]
        gamma /= beta

        # Actualizam elementele coloanei j
        for i in range(r, n):
            A[i, j] -= gamma * u[i]

    # ----------------------------------------------------------
    # TRANSFORMAREA COLOANEI r (o setam la forma dorita)
    # a[r][r] = k  (elementul diagonal devine k)
    # a[i][r] = 0  pentru i > r  (zerouri sub diagonala)
    # ----------------------------------------------------------

    A[r, r] = k
    for i in range(r + 1, n):
        A[i, r] = 0.0

    # ----------------------------------------------------------
    # TRANSFORMAREA VECTORULUI b: b = P_r * b
    #
    # gamma = (u^T * b) / beta = sum_i u[i]*b[i] / beta
    # b[i] = b[i] - gamma * u[i]  pentru i = r..n-1
    # ----------------------------------------------------------

    gamma = 0.0
    for i in range(r, n):
        gamma += u[i] * b[i]
    gamma /= beta

    for i in range(r, n):
        b[i] -= gamma * u[i]

    # ----------------------------------------------------------
    # TRANSFORMAREA MATRICEI Q_bar: Q_bar = P_r * Q_bar
    #
    # Aceleasi calcule ca pentru b, dar pentru fiecare coloana j
    # a matricei Q_bar.
    # La final Q_bar = Q^T
    # ----------------------------------------------------------

    for j in range(n):
        gamma = 0.0
        for i in range(r, n):
            gamma += u[i] * Q_bar[i, j]
        gamma /= beta

        for i in range(r, n):
            Q_bar[i, j] -= gamma * u[i]

# La final, verificam diagonala lui R (care e acum in A)
# Daca vreun element diagonal e aproape 0 -> A e singulara
if not singular:
    for i in range(n):
        if math.fabs(A[i, i]) <= eps:
            singular = True
            print(f"EROARE: R[{i}][{i}] = {A[i,i]} ~ 0. Matricea A este singulara!")
            break

if not singular:
    print("Descompunerea QR Householder calculata cu succes!")
    print(f"R[0][0] = {A[0,0]:.6f}  (element diagonal al lui R)")

# Acum:
# A    = R  (matrice superior triunghiulara)
# Q_bar = Q^T
# b    = Q^T * b_init

# ============================================================
# 4. REZOLVAREA SISTEMULUI Ax = b FOLOSIND QR HOUSEHOLDER
#
# Rx = Q^T * b_init
# b contine deja Q^T * b_init (transformat la pasii de mai sus)
# Rezolvam Rx = b prin substitutie inversa
# ============================================================

def substitutie_inversa(R, rhs, n, eps):
    """
    Rezolva sistemul superior triunghiular R*x = rhs.
    Returneaza vectorul solutie x sau None daca R e singulara.
    """
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if math.fabs(R[i, i]) <= eps:
            print(f"EROARE substitutie inversa: R[{i}][{i}] ~ 0!")
            return None
        s_val = rhs[i]
        for j in range(i + 1, n):
            s_val -= R[i, j] * x[j]
        x[i] = s_val / R[i, i]
    return x

if not singular:
    xHouseholder = substitutie_inversa(A, b, n, eps)
    if xHouseholder is not None:
        print(f"\nxHouseholder (primele {k_display} valori): {xHouseholder[:k_display]}")

# ============================================================
# 5. CALCULUL ERORILOR
#
# a) ||A_init * xHouseholder - b_init||_2
# b) ||A_init * xQR - b_init||_2
# c) ||xHouseholder - s||_2 / ||s||_2
# d) ||xQR - s||_2 / ||s||_2
#
# Toate ar trebui sa fie < 1e-6
# ============================================================

print("\n--- Verificare erori ---")

def norma_euclidiana(v):
    """Calculeaza norma euclidiana (L2) a unui vector."""
    return math.sqrt(sum(x * x for x in v))

def inmultire_matrice_vector(M, v, n):
    """Inmultire matrice-vector, implementata manual."""
    result = np.zeros(n)
    for i in range(n):
        for j in range(n):
            result[i] += M[i, j] * v[j]
    return result

if not singular and xHouseholder is not None:

    # a) Eroarea reziduala pentru xHouseholder
    Ax_H = inmultire_matrice_vector(A_init, xHouseholder, n)
    reziduu_H = norma_euclidiana(Ax_H - b_init)
    print(f"||A_init * xHouseholder - b_init||_2 = {reziduu_H:.6e}  (target < 1e-6)")

    # b) Eroarea reziduala pentru xQR
    Ax_QR = inmultire_matrice_vector(A_init, xQR, n)
    reziduu_QR = norma_euclidiana(Ax_QR - b_init)
    print(f"||A_init * xQR - b_init||_2          = {reziduu_QR:.6e}  (target < 1e-6)")

    # c) Eroarea relativa fata de solutia exacta s, pentru xHouseholder
    norma_s = norma_euclidiana(s)
    eroare_rel_H = norma_euclidiana(xHouseholder - s) / norma_s
    print(f"||xHouseholder - s||_2 / ||s||_2     = {eroare_rel_H:.6e}  (target < 1e-6)")

    # d) Eroarea relativa fata de solutia exacta s, pentru xQR
    eroare_rel_QR = norma_euclidiana(xQR - s) / norma_s
    print(f"||xQR - s||_2 / ||s||_2              = {eroare_rel_QR:.6e}  (target < 1e-6)")

    # Diferenta dintre cele doua solutii
    diff_solutii = norma_euclidiana(xQR - xHouseholder)
    print(f"\n||xQR - xHouseholder||_2             = {diff_solutii:.6e}")

# ============================================================
# 6. CALCULUL INVERSEI FOLOSIND HOUSEHOLDER
#
# Inversa lui A se calculeaza coloana cu coloana.
# Coloana j a inversei = solutia sistemului A*x = e_j
# unde e_j este vectorul cu 1 pe pozitia j si 0 in rest.
#
# Dar avem deja descompunerea QR (A = R acum, Q_bar = Q^T).
# Pentru sistemul A*x = e_j avem:
#   R*x = Q^T * e_j = coloana j din Q^T = linia j din Q
#
# Q^T e stocat in Q_bar, deci coloana j din Q^T = Q_bar[:, j]
# ============================================================

print("\n--- Calculul inversei matricei A ---")

if not singular:
    A_inv_H = np.zeros((n, n))

    for j in range(n):
        # Termenul liber pentru sistemul coloanei j
        # = coloana j din Q^T = Q_bar[:, j]
        rhs_j = Q_bar[:, j].copy()

        # Rezolvam R * x = rhs_j prin substitutie inversa
        # R se afla in matricea A (modificata de Householder)
        col_j = substitutie_inversa(A, rhs_j, n, eps)

        if col_j is None:
            print(f"EROARE la calculul coloanei {j} a inversei!")
            break

        # Stocam solutia in coloana j a inversei
        A_inv_H[:, j] = col_j

    # Calculam inversa cu biblioteca pentru comparatie
    A_inv_lib = np.linalg.inv(A_init)

    # Norma diferentei dintre cele doua inverse
    diff_inv = np.linalg.norm(A_inv_H - A_inv_lib)
    print(f"||A_inv_Householder - A_inv_lib||    = {diff_inv:.6e}")

    if diff_inv < 1e-6:
        print("✓ Inversa calculata corect!")
    else:
        print("✗ Diferenta prea mare la inversa!")

# ============================================================
# 7. TEST PE EXEMPLUL DIN PDF
#    A = [[0,0,4],[1,2,3],[0,1,2]], s = [3,2,1]
#    Solutia exacta e s = [3,2,1]
# ============================================================

print("\n" + "=" * 60)
print("  TEST PE EXEMPLUL DIN PDF (n=3)")
print("=" * 60)

A_ex   = np.array([[0.0, 0.0, 4.0],
                   [1.0, 2.0, 3.0],
                   [0.0, 1.0, 2.0]])
s_ex   = np.array([3.0, 2.0, 1.0])
b_ex   = A_ex @ s_ex          # b = [4, 10, 4]
eps_ex = 1e-8

print(f"b = A*s = {b_ex}  (asteptat [4, 10, 4])")

# Rezolvam cu numpy
x_np = np.linalg.solve(A_ex, b_ex)
print(f"Solutie numpy:       {x_np}  (asteptat [3, 2, 1])")

# Rezolvam cu Householder
A_ex_work = A_ex.copy()
b_ex_work = b_ex.copy()
Q_ex      = np.eye(3)
n_ex      = 3

for r in range(n_ex - 1):
    sigma = sum(A_ex_work[i, r]**2 for i in range(r, n_ex))
    if sigma <= eps_ex:
        break
    k_ex = math.sqrt(sigma)
    if A_ex_work[r, r] > 0:
        k_ex = -k_ex
    beta_ex = sigma - k_ex * A_ex_work[r, r]
    u_ex = np.zeros(n_ex)
    u_ex[r] = A_ex_work[r, r] - k_ex
    for i in range(r + 1, n_ex):
        u_ex[i] = A_ex_work[i, r]

    for j in range(r + 1, n_ex):
        gamma = sum(u_ex[i] * A_ex_work[i, j] for i in range(r, n_ex)) / beta_ex
        for i in range(r, n_ex):
            A_ex_work[i, j] -= gamma * u_ex[i]

    A_ex_work[r, r] = k_ex
    for i in range(r + 1, n_ex):
        A_ex_work[i, r] = 0.0

    gamma = sum(u_ex[i] * b_ex_work[i] for i in range(r, n_ex)) / beta_ex
    for i in range(r, n_ex):
        b_ex_work[i] -= gamma * u_ex[i]

    for j in range(n_ex):
        gamma = sum(u_ex[i] * Q_ex[i, j] for i in range(r, n_ex)) / beta_ex
        for i in range(r, n_ex):
            Q_ex[i, j] -= gamma * u_ex[i]

x_ex = substitutie_inversa(A_ex_work, b_ex_work, n_ex, eps_ex)
print(f"Solutie Householder: {x_ex}  (asteptat [3, 2, 1])")
print(f"Eroare: {np.linalg.norm(x_ex - s_ex):.2e}")

print("\n" + "=" * 60)
print("  PROGRAM TERMINAT")
print("=" * 60)
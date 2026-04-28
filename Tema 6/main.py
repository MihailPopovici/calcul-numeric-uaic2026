"""
TEMA 6 - Aproximarea functiilor prin:
  1. Metoda celor mai mici patrate (polinom Pm)
  2. Functii spline cubice de clasa C2

Ce face programul:
  - Genereaza n+1 puncte xi intre a si b (x0=a, xn=b citite, restul aleatorii)
  - Calculeaza yi = f(xi) folosind o functie data
  - Aproximeaza f(x_bar) in punctul ales x_bar
  - Afiseaza erorile si face grafice
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random

# ─────────────────────────────────────────────
#  FUNCTIA f SI DERIVATA EI (exemple din PDF)
# ─────────────────────────────────────────────

# Exemplul 1 din PDF:  f(x) = x^4 - 12x^3 + 30x^2 + 12
# a=0, b=2, da=0, db=8

def f(x):
    """Functia originala pe care vrem sa o aproximam."""
    return x**4 - 12*x**3 + 30*x**2 + 12

def f_deriv_a():
    """Valoarea derivatei f'(a) = f'(0)"""
    return 0.0   # 4x^3 - 36x^2 + 60x  la x=0 → 0

def f_deriv_b(b):
    """Valoarea derivatei f'(b) = f'(2)"""
    return 4*b**3 - 36*b**2 + 60*b   # = 32 - 144 + 120 = 8 


# ─────────────────────────────────────────────
#  GENERARE NODURI DE INTERPOLARE
# ─────────────────────────────────────────────

def genereaza_noduri(a, b, n):
    """
    Genereaza n+1 noduri: x0=a, xn=b, iar x1..x(n-1) aleatorii in (a,b), sortate.
    
    De ce? Vrem sa acoperim intervalul [a,b] cu puncte cunoscute ale functiei.
    Cu cat avem mai multe puncte, cu atat aproximarea e mai buna.
    """
    # Generam n-1 puncte aleatorii in (a, b)
    interior = sorted(random.uniform(a, b) for _ in range(n - 1))
    noduri = [a] + interior + [b]
    return np.array(noduri)


# ─────────────────────────────────────────────
#  SCHEMA LUI HORNER
# ─────────────────────────────────────────────

def horner(coefs, x0):
    """
    Evalueaza polinomul P(x0) folosind schema lui Horner.
    
    De ce Horner? Este mai eficient decat calculul naiv.
    
    Exemplu: P(x) = 3x^3 + 2x^2 + x + 5 la x0=2
    Horner:  ((3*2 + 2)*2 + 1)*2 + 5 = (8)*2... = 31
    
    coefs = [a_m, a_{m-1}, ..., a_1, a_0]  (de la grad cel mai mare la cel mai mic)
    """
    # Plecam de la coeficientul cel mai mare
    rezultat = coefs[0]
    for c in coefs[1:]:
        rezultat = rezultat * x0 + c
    return rezultat


# ─────────────────────────────────────────────
#  METODA CELOR MAI MICI PATRATE
# ─────────────────────────────────────────────

def least_squares_poly(xi, yi, m):
    """
    Gaseste polinomul Pm(x) de grad m care 'se potriveste cel mai bine'
    cu punctele (xi, yi), in sensul ca minimizeaza suma erorilor la patrat:
    
        min  sum_i (Pm(xi) - yi)^2
    
    Cum functioneaza?
    - Construim o matrice B si un vector f din datele noastre
    - Rezolvam sistemul liniar B*a = f
    - Coeficientii {a0, a1, ..., am} definesc polinomul Pm
    
    De ce "cele mai mici patrate"?  Minimizam suma PATRATELOR erorilor,
    nu erorile in sine (pentru ca erorile negative si pozitive s-ar anula).
    """
    n_plus1 = len(xi)  # numarul de puncte = n+1

    # Construim matricea B de dimensiune (m+1) x (m+1)
    # B[i][j] = sum_k  xi[k]^(i+j)   pentru i,j = 0..m
    B = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            B[i][j] = np.sum(xi ** (i + j))

    # Construim vectorul f de dimensiune m+1
    # f[i] = sum_k  yi[k] * xi[k]^i   pentru i = 0..m
    f_vec = np.zeros(m + 1)
    for i in range(m + 1):
        f_vec[i] = np.sum(yi * xi ** i)

    # Rezolvam sistemul liniar B * a = f
    # numpy.linalg.solve foloseste eliminare Gauss (sau LU) intern
    a = np.linalg.solve(B, f_vec)
    # a = [a0, a1, ..., am]  (de la gradul 0 la gradul m)

    return a


def evalueaza_Pm(a_coefs, x_bar):
    """
    Evalueaza Pm(x_bar) cu schema Horner.
    
    a_coefs = [a0, a1, ..., am]  (de la grad 0 la grad m)
    Schema Horner vrea coeficientii de la grad MARE la mic,
    adica [am, am-1, ..., a1, a0]
    """
    coefs_horner = a_coefs[::-1]   # inversam vectorul
    return horner(coefs_horner, x_bar)


# ─────────────────────────────────────────────
#  FUNCTII SPLINE CUBICE DE CLASA C2
# ─────────────────────────────────────────────

def spline_cubic(xi, yi, da, db):
    """
    Construieste functia spline cubica Sf de clasa C2.
    
    Ce inseamna asta?
    - Pe fiecare interval [xi, xi+1] avem un polinom cubic (grad 3)
    - La punctele de jonctiune, polinomii "se lipesc" lin:
        * Sf e continua (fara salturi)
        * Sf' e continua (fara colturi)  
        * Sf'' e continua (fara schimbari bruste de curbura) → de aici "C2"
    - In plus, derivatele la capete coincid cu f'(a) si f'(b)
    
    Rezulta un sistem liniar HA = f din care gasim "momentele" Ai.
    Momentele Ai controleaza cat de curbata e spline-ul in fiecare nod.
    """
    n = len(xi) - 1  # numarul de intervale

    # hi = lungimea intervalului i: hi = xi[i+1] - xi[i]
    h = np.diff(xi)   # h[i] = xi[i+1] - xi[i]

    # ── Construim matricea H (tridiagonala, dimensiune (n+1) x (n+1)) ──
    H = np.zeros((n + 1, n + 1))

    # Prima linie: 2*h0*A0 + h0*A1 = ...
    H[0, 0] = 2 * h[0]
    H[0, 1] = h[0]

    # Liniile interioare i=1..n-1
    for i in range(1, n):
        H[i, i - 1] = h[i - 1]
        H[i, i]     = 2 * (h[i - 1] + h[i])
        H[i, i + 1] = h[i]

    # Ultima linie: h_{n-1}*A_{n-1} + 2*h_{n-1}*A_n = ...
    H[n, n - 1] = h[n - 1]
    H[n, n]     = 2 * h[n - 1]

    # ── Construim vectorul f ──
    f_vec = np.zeros(n + 1)

    # Prima componenta
    f_vec[0] = 6 * ((yi[1] - yi[0]) / h[0] - da)

    # Componentele interioare
    for i in range(1, n):
        f_vec[i] = 6 * ((yi[i + 1] - yi[i]) / h[i] - (yi[i] - yi[i - 1]) / h[i - 1])

    # Ultima componenta
    f_vec[n] = 6 * (db - (yi[n] - yi[n - 1]) / h[n - 1])

    # ── Rezolvam HA = f ──
    A = np.linalg.solve(H, f_vec)
    # A[i] sunt "momentele" spline-ului in fiecare nod

    return A, h


def evalueaza_spline(xi, yi, A, h, x_bar):
    """
    Evalueaza Sf(x_bar) gasind mai intai intervalul [xi0, xi0+1] care contine x_bar.
    Aplica apoi formula cubica din PDF.
    """
    # Gasim i0 astfel incat x_bar in [xi[i0], xi[i0+1]]
    i0 = np.searchsorted(xi, x_bar, side='right') - 1
    # Limitam la intervalul valid [0, n-1]
    i0 = int(np.clip(i0, 0, len(xi) - 2))

    hi0  = h[i0]
    Ai0  = A[i0]
    Ai1  = A[i0 + 1]
    xi0  = xi[i0]
    xi1  = xi[i0 + 1]
    yi0  = yi[i0]
    yi1  = yi[i0 + 1]

    # Coeficientii bi si ci din formula spline
    bi = (yi1 - yi0) / hi0 - hi0 * (Ai1 - Ai0) / 6
    ci = (xi1 * yi0 - xi0 * yi1) / hi0 - hi0 * (xi1 * Ai0 - xi0 * Ai1) / 6

    # Formula Sf(x_bar):
    val = (
        (x_bar - xi0)**3 * Ai1 / (6 * hi0)
        + (xi1 - x_bar)**3 * Ai0 / (6 * hi0)
        + bi * x_bar
        + ci
    )
    return val


# ─────────────────────────────────────────────
#  PROGRAMUL PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TEMA 6 – Aproximarea functiilor")
    print("  1. Metoda celor mai mici patrate (polinom)")
    print("  2. Functii spline cubice de clasa C2")
    print("=" * 60)

    # ── Parametri de intrare ──
    a = 0.0         # x0 = a (capatul stang al intervalului)
    b = 2.0         # xn = b (capatul drept)
    n = 8           # vom avea n+1 = 9 noduri (0..n)
    x_bar = 1.2     # punctul in care vrem sa aproximam f
    m = 3           # gradul polinomului Pm (< 6, conform cerintei)

    da = f_deriv_a()       # f'(a)
    db = f_deriv_b(b)      # f'(b)

    print(f"\nParametri:")
    print(f"  a = {a}, b = {b}")
    print(f"  n+1 = {n+1} noduri, grad polinom m = {m}")
    print(f"  x_bar = {x_bar}  (punctul de aproximat)")
    print(f"  da = f'({a}) = {da},  db = f'({b}) = {db}")
    print(f"  Valoarea exacta f({x_bar}) = {f(x_bar):.6f}")

    # Fixam seed-ul pentru reproducibilitate (comenteaza pentru noduri aleatorii diferite)
    random.seed(42)

    # ── Generare noduri ──
    xi = genereaza_noduri(a, b, n)
    yi = np.array([f(x) for x in xi])

    print(f"\nNodurile generate (xi):")
    for i, (x, y) in enumerate(zip(xi, yi)):
        print(f"  x{i} = {x:.4f}   f(x{i}) = {y:.4f}")

    # ─────────────────────────────
    #  METODA 1: CELE MAI MICI PATRATE
    # ─────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  METODA 1: Polinomul de gradul {m} (Least Squares)")
    print(f"{'─'*50}")

    # Calculam coeficientii polinomului
    a_coefs = least_squares_poly(xi, yi, m)
    print(f"\nCoeficientii Pm (a0, a1, ..., a{m}):")
    for i, ai in enumerate(a_coefs):
        print(f"  a{i} = {ai:.6f}")

    # Evaluam Pm(x_bar) cu schema Horner
    Pm_xbar = evalueaza_Pm(a_coefs, x_bar)
    f_xbar  = f(x_bar)
    eroare_Pm = abs(Pm_xbar - f_xbar)

    # Suma erorilor la toate nodurile: sum |Pm(xi) - yi|
    suma_erori = sum(abs(evalueaza_Pm(a_coefs, x) - y) for x, y in zip(xi, yi))

    print(f"\nRezultate Pm:")
    print(f"  Pm({x_bar}) = {Pm_xbar:.6f}")
    print(f"  f({x_bar})  = {f_xbar:.6f}  (valoare exacta)")
    print(f"  |Pm({x_bar}) - f({x_bar})| = {eroare_Pm:.6f}  ← eroarea de aproximare")
    print(f"  Σ|Pm(xi) - yi| = {suma_erori:.6f}  ← eroarea totala pe noduri")

    # Testam cu mai multe grade m pentru comparatie
    print(f"\n  Comparatie erori pentru diferite grade m:")
    print(f"  {'m':>4} | {'Pm(x_bar)':>14} | {'Eroare punct':>14} | {'Eroare totala':>14}")
    print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    for grad in range(1, 6):
        try:
            ac = least_squares_poly(xi, yi, grad)
            val = evalueaza_Pm(ac, x_bar)
            err_punct = abs(val - f_xbar)
            err_total = sum(abs(evalueaza_Pm(ac, x) - y) for x, y in zip(xi, yi))
            print(f"  {grad:>4} | {val:>14.6f} | {err_punct:>14.6f} | {err_total:>14.6f}")
        except np.linalg.LinAlgError:
            print(f"  {grad:>4} | Sistem singular - nu se poate rezolva")

    # ─────────────────────────────
    #  METODA 2: SPLINE CUBIC
    # ─────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  METODA 2: Spline cubic C2")
    print(f"{'─'*50}")

    # Calculam momentele A
    A, h = spline_cubic(xi, yi, da, db)

    print(f"\nMomentele Ai ale spline-ului:")
    for i, Ai in enumerate(A):
        print(f"  A{i} = {Ai:.6f}")

    # Evaluam Sf(x_bar)
    Sf_xbar = evalueaza_spline(xi, yi, A, h, x_bar)
    eroare_Sf = abs(Sf_xbar - f_xbar)

    print(f"\nRezultate Sf:")
    print(f"  Sf({x_bar}) = {Sf_xbar:.6f}")
    print(f"  f({x_bar})  = {f_xbar:.6f}  (valoare exacta)")
    print(f"  |Sf({x_bar}) - f({x_bar})| = {eroare_Sf:.6f}  ← eroarea de aproximare")

    # ─────────────────────────────
    #  GRAFICE
    # ─────────────────────────────
    print(f"\n{'─'*50}")
    print("  Generare grafice...")
    print(f"{'─'*50}")

    x_plot = np.linspace(a, b, 500)      # 500 puncte pentru curbe netede
    y_exact = np.array([f(x) for x in x_plot])

    # Pm(x) pe tot intervalul, cu gradul m ales
    y_Pm = np.array([evalueaza_Pm(a_coefs, x) for x in x_plot])

    # Sf(x) pe tot intervalul
    y_Sf = np.array([evalueaza_spline(xi, yi, A, h, x) for x in x_plot])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Tema 6 – Aproximarea functiei f(x) = x⁴ - 12x³ + 30x² + 12\n"
                 f"pe [{a}, {b}] cu {n+1} noduri", fontsize=13, fontweight='bold')

    # ── Graficul 1: toate cele 3 curbe ──
    ax1 = axes[0]
    ax1.plot(x_plot, y_exact, 'k-',  linewidth=2.5, label='f(x) exacta', zorder=5)
    ax1.plot(x_plot, y_Pm,    'b--', linewidth=2,   label=f'Pm(x) grad {m} (cele mai mici patrate)')
    ax1.plot(x_plot, y_Sf,    'r-.',  linewidth=2,   label='Sf(x) (spline cubic)')
    ax1.scatter(xi, yi, color='green', s=80, zorder=6, label='Noduri (xi, yi)')
    ax1.axvline(x=x_bar, color='purple', linestyle=':', alpha=0.7, label=f'x̄ = {x_bar}')
    ax1.scatter([x_bar], [f_xbar], color='purple', s=120, zorder=7, marker='*',
                label=f'f(x̄) = {f_xbar:.4f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Comparatie f(x), Pm(x) si Sf(x)')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ── Graficul 2: erorile celor doua metode ──
    ax2 = axes[1]
    eroare_Pm_plot = np.abs(y_Pm - y_exact)
    eroare_Sf_plot = np.abs(y_Sf - y_exact)
    ax2.plot(x_plot, eroare_Pm_plot, 'b--', linewidth=2, label=f'|Pm(x) - f(x)|')
    ax2.plot(x_plot, eroare_Sf_plot, 'r-.',  linewidth=2, label='|Sf(x) - f(x)|')
    ax2.axvline(x=x_bar, color='purple', linestyle=':', alpha=0.7, label=f'x̄ = {x_bar}')
    ax2.scatter([x_bar], [eroare_Pm], color='blue', s=100, zorder=5,
                label=f'Eroare Pm(x̄) = {eroare_Pm:.4f}')
    ax2.scatter([x_bar], [eroare_Sf], color='red', s=100, zorder=5,
                label=f'Eroare Sf(x̄) = {eroare_Sf:.4f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Eroare absoluta')
    ax2.set_title('Erorile absolute ale celor doua metode')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')   # scala logaritmica ca sa vedem mai bine diferentele

    output_path = os.path.join(os.path.dirname(__file__), "tema6_grafice.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafice salvate: {output_path}")

    print(f"\n{'='*60}")
    print("  REZUMAT FINAL")
    print(f"{'='*60}")
    print(f"  Valoare exacta:        f({x_bar}) = {f_xbar:.6f}")
    print(f"  Polinom grad {m}:        Pm({x_bar}) = {Pm_xbar:.6f}   "
          f"eroare = {eroare_Pm:.6f}")
    print(f"  Spline cubic:          Sf({x_bar}) = {Sf_xbar:.6f}   "
          f"eroare = {eroare_Sf:.6f}")
    metoda_mai_buna = "Spline cubic" if eroare_Sf < eroare_Pm else f"Polinom grad {m}"
    print(f"\n  ➜ Metoda mai precisa in x̄={x_bar}: {metoda_mai_buna}")


if __name__ == "__main__":
    main()
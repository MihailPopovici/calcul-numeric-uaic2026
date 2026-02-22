"""
Problema 3 - Aproximarea functiei tangenta
==========================================
Doua metode: fractii continue (Lentz) si polinom MacLaurin.
"""

import math
import time
import random


# ==============================================================================
# METODA 1: Fractii continue – algoritmul Lentz modificat
# ==============================================================================
#
# Enuntul scrie tan ca fractie continua in notatia compacta:
#
#   tan x = x/(1+)  (-x^2)/(3+)  (-x^2)/(5+)  (-x^2)/(7+)  ...
#
# Forma standard a unei fractii continue este b0 + a1/(b1 + a2/(b2 + ...)).
# Ca sa punem tan in aceasta forma, scriem:
#
#   b0 = 0  (termenul liber e zero, tan(0)=0)
#   a1 = x,    b1 = 1
#   a2 = -x^2, b2 = 3
#   a3 = -x^2, b3 = 5   ...si asa mai departe
#
# Pe scurt: aj = -x^2 pentru j >= 2, bj = 2j-1 pentru j >= 1, a1 = x.

def my_tan_cf(x, eps=1e-12):
    """
    Calculeaza tan(x) prin fractii continue cu algoritmul Lentz modificat.
    x trebuie sa fie in (-pi/2, pi/2).
    """
    mic = 1e-30  # valoare minuscula ca sa evitam impartirea la zero

    x2 = x * x

    # b0 = 0, dar zero se inlocuieste cu mic ca sa nu impartim la 0
    f = mic
    C = f
    D = 0.0

    j = 1
    while True:
        a_j = x   if j == 1 else -x2   # primul termen e x, restul -x^2
        b_j = 2 * j - 1                 # 1, 3, 5, 7, ...

        D = b_j + a_j * D
        if D == 0.0:
            D = mic
        D = 1.0 / D

        C = b_j + a_j / C
        if C == 0.0:
            C = mic

        delta = C * D
        f = delta * f

        if abs(delta - 1.0) < eps:
            break
        j += 1

    return f  # rezultatul este direct tan(x)


# ==============================================================================
# METODA 2: Polinom MacLaurin (grad 9)
# ==============================================================================

C1 = 1.0 / 3.0       # coeficient x^3
C2 = 2.0 / 15.0      # coeficient x^5
C3 = 17.0 / 315.0    # coeficient x^7
C4 = 62.0 / 2835.0   # coeficient x^9

def poly_core(x):
    """Polinomul MacLaurin propriu-zis, valid pe (-pi/4, pi/4)."""
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x6 = x4 * x2
    return x + x3 * (C1 + C2 * x2 + C3 * x4 + C4 * x6)

def my_tan_poly(x):
    """
    Calculeaza tan(x) prin polinom MacLaurin.
    x trebuie sa fie in (-pi/2, pi/2).
    """
    PI4 = math.pi / 4.0
    PI2 = math.pi / 2.0

    if abs(x) <= PI4:
        return poly_core(x)
    else:
        # tan(x) = 1/tan(pi/2 - x), iar (pi/2 - |x|) e in (0, pi/4)
        reduced = PI2 - abs(x)
        result  = 1.0 / poly_core(reduced)
        return result if x > 0 else -result

# ==============================================================================
# COMPARATIE pe 10.000 de valori aleatoare din (-pi/2, pi/2)
# ==============================================================================

def compare_10000(n=10000, eps=1e-12):
    half_pi = math.pi / 2.0
    xs = [random.uniform(-half_pi, half_pi) for _ in range(n)]

    t0       = time.perf_counter()
    errs_cf  = [abs(math.tan(x) - my_tan_cf(x, eps)) for x in xs]
    t_cf     = time.perf_counter() - t0

    t0        = time.perf_counter()
    errs_poly = [abs(math.tan(x) - my_tan_poly(x)) for x in xs]
    t_poly    = time.perf_counter() - t0

    print("=" * 60)
    print(f"COMPARATIE PE {n} VALORI ALEATOARE DIN (-pi/2, pi/2)")
    print("=" * 60)
    print(f"{'Metoda':<22} {'Eroare max':>12} {'Eroare medie':>14} {'Timp (s)':>9}")
    print("-" * 60)
    print(f"{'Fractii continue':<22} {max(errs_cf):>12.2e} {sum(errs_cf)/n:>14.2e} {t_cf:>9.4f}")
    print(f"{'Polinom MacLaurin':<22} {max(errs_poly):>12.2e} {sum(errs_poly)/n:>14.2e} {t_poly:>9.4f}")
    print("=" * 60)
    print()
    print("Concluzie:")
    if max(errs_cf) < max(errs_poly):
        print("  Fractiile continue sunt mai precise.")
    else:
        print("  Polinomul MacLaurin este mai precis.")
    if t_cf < t_poly:
        print("  Fractiile continue sunt mai rapide.")
    else:
        print("  Polinomul MacLaurin este mai rapid.")


if __name__ == "__main__":
    compare_10000(n=10000, eps=1e-12)
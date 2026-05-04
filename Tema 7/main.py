"""
TEMA 7 - Aproximarea radacinilor unui polinom cu:
  1. Metoda lui Newton
  2. Metoda lui Olver

Ce face programul:
  - Calculeaza intervalul [-R, R] in care se afla toate radacinile reale
  - Pornind din mai multe puncte x0 aleatorii din [-R, R], aplica ambele metode
  - Compara cele doua metode: numarul de pasi pana la aceeasi precizie
  - Afiseaza radacinile distincte gasite si le salveaza in fisier
  - Foloseste schema lui Horner pentru evaluarea polinomului si derivatelor
"""

import numpy as np
import random
import os

# ─────────────────────────────────────────────
#  POLINOMUL (alege unul dintre exemple din PDF)
# ─────────────────────────────────────────────

# Exemplu 1: P(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
# Radacini exacte: 1, 2, 3
COEFS = [1.0, -6.0, 11.0, -6.0]
RADACINI_EXACTE = [1.0, 2.0, 3.0]
NUME_POLINOM = "P(x) = x³ - 6x² + 11x - 6  (radacini exacte: 1, 2, 3)"

# Exemplu 2 (decomentati pentru a schimba):
# P(x) = 42x^4 - 55x^3 - 42x^2 + 49x - 6
# Radacini: 2/3, 1/7, -1, 3/2
# COEFS = [42.0, -55.0, -42.0, 49.0, -6.0]
# RADACINI_EXACTE = [2/3, 1/7, -1.0, 3/2]
# NUME_POLINOM = "P(x) = 42x⁴ - 55x³ - 42x² + 49x - 6"

# Exemplu 3:
# P(x) = 8x^4 - 38x^3 + 49x^2 - 22x + 3
# Radacini: 1, 1/2, 3, 1/4
# COEFS = [8.0, -38.0, 49.0, -22.0, 3.0]
# RADACINI_EXACTE = [1.0, 0.5, 3.0, 0.25]
# NUME_POLINOM = "P(x) = 8x⁴ - 38x³ + 49x² - 22x + 3"

# Parametri
EPSILON = 1e-8    # precizia dorita
K_MAX   = 1000    # numarul maxim de iteratii (protectie la divergenta)


# ─────────────────────────────────────────────
#  SCHEMA LUI HORNER (evaluare P, P', P'')
# ─────────────────────────────────────────────

def horner_P(a, v):
    """
    Calculeaza P(v) folosind schema lui Horner.

    Ideea: in loc de a calcula fiecare termen separat (lent),
    grupeaza calculele astfel: P(x) = (...((a0*x + a1)*x + a2)*x + ...)*x + an
    Rezulta acelasi raspuns dar cu mult mai putine inmultiri.

    Parametri:
      a = lista coeficientilor [a0, a1, ..., an]  (a0 = coef. grad cel mai mare)
      v = punctul in care evaluam
    Returneaza:
      P(v) = valoarea polinomului in v
    """
    b = a[0]
    for i in range(1, len(a)):
        b = b * v + a[i]
    return b


def horner_P_dP(a, v):
    """
    Calculeaza simultan P(v) si P'(v) cu schema Horner extinsa.

    Cum? Dupa prima trecere Horner obtinem coeficientii polinomului Q
    (catul impartirii P(x) / (x - v)). O a doua trecere Horner pe Q
    in acelasi punct v da P'(v).

    Returneaza: (P(v), P'(v))
    """
    n = len(a) - 1   # gradul polinomului

    # Prima trecere: calculeaza P(v) si coeficientii lui Q
    b = a[0]
    b_coefs = [b]
    for i in range(1, n + 1):
        b = b * v + a[i]
        b_coefs.append(b)
    Pv = b_coefs[n]   # P(v) = ultimul b

    # A doua trecere: calculeaza P'(v) din coeficientii Q = b_coefs[0..n-1]
    c = b_coefs[0]
    for i in range(1, n):
        c = c * v + b_coefs[i]
    dPv = c   # P'(v)

    return Pv, dPv


def horner_P_dP_d2P(a, v):
    """
    Calculeaza simultan P(v), P'(v) si P''(v) cu 3 treceri Horner.

    Necesare pentru metoda Olver care are nevoie si de derivata a doua.

    Returneaza: (P(v), P'(v), P''(v))
    """
    n = len(a) - 1

    # Prima trecere → P(v) + coeficientii Q1
    b = a[0]
    b_coefs = [b]
    for i in range(1, n + 1):
        b = b * v + a[i]
        b_coefs.append(b)
    Pv = b_coefs[n]

    # A doua trecere pe Q1 → P'(v) + coeficientii Q2
    c = b_coefs[0]
    c_coefs = [c]
    for i in range(1, n):
        c = c * v + b_coefs[i]
        c_coefs.append(c)
    dPv = c_coefs[n - 1]

    # A treia trecere pe Q2 → P''(v) / 2  (de fapt obtinem P''(v)/2!)
    # Corect: P''(v) = 2 * (ultima valoare din a 3-a trecere Horner)
    if n >= 2:
        d = c_coefs[0]
        for i in range(1, n - 1):
            d = d * v + c_coefs[i]
        d2Pv = 2.0 * d
    else:
        d2Pv = 0.0

    return Pv, dPv, d2Pv


# ─────────────────────────────────────────────
#  INTERVALUL [-R, R] AL RADACINILOR
# ─────────────────────────────────────────────

def calculeaza_R(a):
    """
    Toate radacinile reale ale polinomului P se afla in [-R, R] unde:
        R = (|a0| + A) / |a0|,   A = max(|a1|, |a2|, ..., |an|)

    De ce? Aceasta e o margine superioara clasica pentru modulul radacinilor.
    Nu e cea mai stransa posibila, dar e simplu de calculat si garantata.
    """
    a0 = a[0]
    A = max(abs(ai) for ai in a[1:])
    R = (abs(a0) + A) / abs(a0)
    return R


# ─────────────────────────────────────────────
#  METODA NEWTON
# ─────────────────────────────────────────────

def metoda_newton(a, x0, eps=EPSILON, kmax=K_MAX):
    """
    Metoda Newton (Newton-Raphson) de gasire a radacinilor.

    Idee geometrica: la fiecare pas, duci tangenta la curba P(x)
    in punctul (xk, P(xk)) si gasesti unde taie axa Ox → acesta e xk+1.
    Formula: xk+1 = xk - P(xk) / P'(xk)

    Converge rapid (ordin 2) cand x0 e aproape de radacina,
    dar poate diverge daca P'(xk) e aproape de 0.

    Returneaza: (radacina_gasita, nr_pasi, converge_boolean)
    """
    x = x0
    for k in range(1, kmax + 1):
        Pv, dPv = horner_P_dP(a, x)

        # Daca derivata e prea mica, nu putem imparti → oprim
        if abs(dPv) <= eps:
            return x, k, False

        delta_x = Pv / dPv   # pasul Newton

        x = x - delta_x

        # Criteriul de oprire: pasul a devenit suficient de mic
        if abs(delta_x) < eps:
            return x, k, True

        # Protectie la divergenta: daca pasul e enorm, diverge
        if abs(delta_x) > 1e8:
            return x, k, False

    return x, kmax, False


# ─────────────────────────────────────────────
#  METODA OLVER
# ─────────────────────────────────────────────

def metoda_olver(a, x0, eps=EPSILON, kmax=K_MAX):
    """
    Metoda Olver — o imbunatatire a metodei Newton.

    Formula: xk+1 = xk - Delta_x
    unde: Delta_x = P(xk)/P'(xk) + (1/2) * ck
          ck = [P(xk)]^2 * P''(xk) / [P'(xk)]^3

    Termenul in plus fata de Newton (1/2 * ck) 'corecteaza' directia
    tinand cont si de curbura functiei (P''). Rezultatul: converge
    in mai putini pasi decat Newton (ordin 3 fata de ordin 2).

    Returneaza: (radacina_gasita, nr_pasi, converge_boolean)
    """
    x = x0
    for k in range(1, kmax + 1):
        Pv, dPv, d2Pv = horner_P_dP_d2P(a, x)

        if abs(dPv) <= eps:
            return x, k, False

        # Termenul Newton de baza
        newton_term = Pv / dPv

        # Termenul de corectie specific Olver
        ck = (Pv ** 2 * d2Pv) / (dPv ** 3)
        delta_x = newton_term + 0.5 * ck

        x = x - delta_x

        if abs(delta_x) < eps:
            return x, k, True

        if abs(delta_x) > 1e8:
            return x, k, False

    return x, kmax, False


# ─────────────────────────────────────────────
#  GESTIONAREA RADACINILOR DISTINCTE
# ─────────────────────────────────────────────

def este_radacina_noua(radacina, lista_existente, eps=EPSILON):
    """
    Verifica daca o valoare gasita e o radacina noua (distincta de cele deja gasite).
    Doua valori sunt considerate IDENTICE daca diferenta lor absoluta < eps.
    """
    for r in lista_existente:
        if abs(radacina - r) <= eps:
            return False   # deja avem aceasta radacina
    return True


def este_radacina_valida(a, x, eps=1e-4):
    """
    Verifica daca x este intr-adevar aproape de o radacina: |P(x)| trebuie sa fie mica.
    Evita sa 'acceptam' valori divergente ca radacini.
    """
    return abs(horner_P(a, x)) < eps


# ─────────────────────────────────────────────
#  PROGRAMUL PRINCIPAL
# ─────────────────────────────────────────────

def main():
    a = COEFS
    n = len(a) - 1   # gradul polinomului

    separator = "=" * 65

    print(separator)
    print("  TEMA 7 – Aproximarea radacinilor unui polinom")
    print("  Metoda Newton vs. Metoda Olver")
    print(separator)
    print(f"\n  Polinom: {NUME_POLINOM}")
    print(f"  Grad: {n}")
    print(f"  Coeficienti: {a}")
    print(f"  Precizie epsilon = {EPSILON}")

    # ── Pasul 1: Intervalul radacinilor ──
    R = calculeaza_R(a)
    print(f"\n{'─'*65}")
    print(f"  PASUL 1: Intervalul radacinilor")
    print(f"{'─'*65}")
    print(f"  A = max|ai| = {max(abs(ai) for ai in a[1:])}")
    print(f"  R = (|a0| + A) / |a0| = {R:.4f}")
    print(f"  => Toate radacinile reale se afla in [{-R:.4f}, {R:.4f}]")

    # ── Pasul 2: Puncte de start x0 distribuite uniform in [-R, R] ──
    random.seed(7)
    NR_PUNCTE_START = 30    # incercam 30 de puncte de start diferite
    puncte_start = [random.uniform(-R, R) for _ in range(NR_PUNCTE_START)]
    # Adaugam si cateva puncte fixe strategice (capete, mijloc, sfert)
    puncte_start += list(np.linspace(-R, R, 20))

    print(f"\n  Incercam {len(puncte_start)} puncte de start x0 in [{-R:.2f}, {R:.2f}]")

    # ── Pasul 3: Aplicam ambele metode ──
    radacini_newton = []   # radacinile gasite de Newton  [(valoare, pasi), ...]
    radacini_olver  = []   # radacinile gasite de Olver

    stats_newton = []   # (x0, radacina, pasi, converge) pentru toate punctele start
    stats_olver  = []

    for x0 in puncte_start:
        # Newton
        r_n, pasi_n, ok_n = metoda_newton(a, x0)
        stats_newton.append((x0, r_n, pasi_n, ok_n))
        if ok_n and este_radacina_valida(a, r_n):
            if este_radacina_noua(r_n, [r for r, _ in radacini_newton]):
                radacini_newton.append((r_n, pasi_n))

        # Olver
        r_o, pasi_o, ok_o = metoda_olver(a, x0)
        stats_olver.append((x0, r_o, pasi_o, ok_o))
        if ok_o and este_radacina_valida(a, r_o):
            if este_radacina_noua(r_o, [r for r, _ in radacini_olver]):
                radacini_olver.append((r_o, pasi_o))

    # Sortam dupa valoare
    radacini_newton.sort(key=lambda t: t[0])
    radacini_olver.sort(key=lambda t: t[0])

    # ── Pasul 4: Afisare rezultate ──
    print(f"\n{'─'*65}")
    print(f"  REZULTATE: Radacini gasite")
    print(f"{'─'*65}")

    print(f"\n  [NEWTON]  {len(radacini_newton)} radacini distincte gasite:")
    print(f"  {'Radacina':>16} | {'Pasi':>6} | {'P(radacina)':>14} | {'Eroare fata de exacta':>20}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*14}-+-{'-'*20}")
    for r, pasi in radacini_newton:
        Pr = horner_P(a, r)
        # Gasim cea mai apropiata radacina exacta
        eroare = min(abs(r - re) for re in RADACINI_EXACTE)
        print(f"  {r:>16.10f} | {pasi:>6} | {Pr:>14.2e} | {eroare:>20.2e}")

    print(f"\n  [OLVER]   {len(radacini_olver)} radacini distincte gasite:")
    print(f"  {'Radacina':>16} | {'Pasi':>6} | {'P(radacina)':>14} | {'Eroare fata de exacta':>20}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*14}-+-{'-'*20}")
    for r, pasi in radacini_olver:
        Pr = horner_P(a, r)
        eroare = min(abs(r - re) for re in RADACINI_EXACTE)
        print(f"  {r:>16.10f} | {pasi:>6} | {Pr:>14.2e} | {eroare:>20.2e}")

    # ── Pasul 5: Comparatie Newton vs Olver ──
    print(f"\n{'─'*65}")
    print(f"  COMPARATIE: Newton vs. Olver (pentru radacinile comune)")
    print(f"{'─'*65}")

    # Gasim radacini comune (gasite de ambele) si comparam pasii
    print(f"\n  {'Radacina exacta':>16} | {'Pasi Newton':>12} | {'Pasi Olver':>11} | {'Avantaj':>10}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*11}-+-{'-'*10}")
    for re in sorted(RADACINI_EXACTE):
        # Gasim cat de multi pasi a folosit Newton pentru aceasta radacina
        pasi_n_list = [pasi for r, pasi in radacini_newton if abs(r - re) < 1e-4]
        pasi_o_list = [pasi for r, pasi in radacini_olver  if abs(r - re) < 1e-4]
        pn = min(pasi_n_list) if pasi_n_list else "—"
        po = min(pasi_o_list) if pasi_o_list else "—"
        if isinstance(pn, int) and isinstance(po, int):
            avantaj = "Olver" if po < pn else ("Newton" if pn < po else "Egal")
        else:
            avantaj = "—"
        print(f"  {re:>16.6f} | {str(pn):>12} | {str(po):>11} | {avantaj:>10}")

    print(f"\n  Explicatie: Olver are ordin de convergenta 3 (cubic),")
    print(f"  Newton are ordin 2 (patratic) => Olver converge in mai putini pasi.")

    # ── Pasul 6: Detaliu pas-cu-pas pentru un x0 ales ──
    print(f"\n{'─'*65}")
    print(f"  DETALIU PAS-CU-PAS (pentru x0 = {-R/2:.2f})")
    print(f"{'─'*65}")

    x0_demo = -R / 2
    # Cautam cea mai apropiata radacina de x0_demo ca sa fie relevant
    x0_demo = sorted(puncte_start, key=lambda x: abs(horner_P(a, x)))[0]

    def newton_verbose(a, x0, eps=EPSILON, kmax=30):
        x = x0
        rows = []
        for k in range(1, kmax + 1):
            Pv, dPv = horner_P_dP(a, x)
            if abs(dPv) <= eps:
                break
            delta = Pv / dPv
            x_nou = x - delta
            rows.append((k, x, Pv, dPv, delta, x_nou))
            x = x_nou
            if abs(delta) < eps:
                break
        return rows

    def olver_verbose(a, x0, eps=EPSILON, kmax=30):
        x = x0
        rows = []
        for k in range(1, kmax + 1):
            Pv, dPv, d2Pv = horner_P_dP_d2P(a, x)
            if abs(dPv) <= eps:
                break
            newton_t = Pv / dPv
            ck = (Pv**2 * d2Pv) / (dPv**3)
            delta = newton_t + 0.5 * ck
            x_nou = x - delta
            rows.append((k, x, Pv, dPv, delta, x_nou))
            x = x_nou
            if abs(delta) < eps:
                break
        return rows

    # Alegem x0 aproape de prima radacina
    x0_demo = RADACINI_EXACTE[0] + 0.5   # putin la dreapta primei radacini

    rows_n = newton_verbose(a, x0_demo)
    rows_o = olver_verbose(a, x0_demo)

    print(f"\n  x0 = {x0_demo}  (radacina cautata ≈ {RADACINI_EXACTE[0]})")
    print(f"\n  Newton ({len(rows_n)} pasi):")
    print(f"  {'k':>3} | {'xk':>14} | {'P(xk)':>12} | {'Delta_x':>12}")
    print(f"  {'-'*3}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")
    for k, x, Pv, dPv, delta, x_nou in rows_n:
        print(f"  {k:>3} | {x:>14.8f} | {Pv:>12.2e} | {delta:>12.2e}")

    print(f"\n  Olver ({len(rows_o)} pasi):")
    print(f"  {'k':>3} | {'xk':>14} | {'P(xk)':>12} | {'Delta_x':>12}")
    print(f"  {'-'*3}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")
    for k, x, Pv, dPv, delta, x_nou in rows_o:
        print(f"  {k:>3} | {x:>14.8f} | {Pv:>12.2e} | {delta:>12.2e}")

    # ── Pasul 7: Salvare in fisier ──
    output_dir = os.path.dirname(__file__)
    output_txt_path = os.path.join(output_dir, "tema7_radacini.txt")
    with open(output_txt_path, "w", encoding="utf-8") as fout:
        fout.write("TEMA 7 – Radacinile polinomului\n")
        fout.write(f"Polinom: {NUME_POLINOM}\n")
        fout.write(f"Precizie: epsilon = {EPSILON}\n\n")

        fout.write("=" * 55 + "\n")
        fout.write("RADACINI DISTINCTE GASITE (Newton)\n")
        fout.write("=" * 55 + "\n")
        for r, pasi in radacini_newton:
            fout.write(f"  x* = {r:.10f}   (convergenta in {pasi} pasi)\n")

        fout.write("\n" + "=" * 55 + "\n")
        fout.write("RADACINI DISTINCTE GASITE (Olver)\n")
        fout.write("=" * 55 + "\n")
        for r, pasi in radacini_olver:
            fout.write(f"  x* = {r:.10f}   (convergenta in {pasi} pasi)\n")

        fout.write("\n" + "=" * 55 + "\n")
        fout.write("COMPARATIE PASI Newton vs Olver\n")
        fout.write("=" * 55 + "\n")
        for re in sorted(RADACINI_EXACTE):
            pasi_n_list = [pasi for r, pasi in radacini_newton if abs(r - re) < 1e-4]
            pasi_o_list = [pasi for r, pasi in radacini_olver  if abs(r - re) < 1e-4]
            pn = min(pasi_n_list) if pasi_n_list else "—"
            po = min(pasi_o_list) if pasi_o_list else "—"
            fout.write(f"  Radacina {re:.4f}: Newton={pn} pasi, Olver={po} pasi\n")

    print(f"\n  Rezultate salvate in: {output_txt_path}")

    # ── Pasul 8: Grafice ──
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Tema 7 – Radacinile polinomului\n{NUME_POLINOM}",
                 fontsize=12, fontweight='bold')

    # ─ Graficul 1: Polinomul si radacinile gasite ─
    ax1 = axes[0]
    x_plot = np.linspace(-R - 0.5, R + 0.5, 800)
    y_plot = np.array([horner_P(a, x) for x in x_plot])

    ax1.plot(x_plot, y_plot, 'k-', linewidth=2, label='P(x)', zorder=3)
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax1.axvline(-R, color='orange', linewidth=1.5, linestyle=':', alpha=0.8, label=f'[-R, R] = [{-R:.2f}, {R:.2f}]')
    ax1.axvline( R, color='orange', linewidth=1.5, linestyle=':', alpha=0.8)

    # Radacinile exacte
    for re in RADACINI_EXACTE:
        ax1.scatter([re], [0], color='black', s=120, zorder=6, marker='D')

    # Radacinile gasite de Newton (albastru)
    for r, _ in radacini_newton:
        ax1.scatter([r], [0], color='blue', s=80, zorder=5, alpha=0.8)

    # Radacinile gasite de Olver (rosu)
    for r, _ in radacini_olver:
        ax1.scatter([r], [0.0], color='red', s=50, zorder=4, marker='^', alpha=0.8)

    patch_exact  = mpatches.Patch(color='black', label='Radacini exacte (♦)')
    patch_newton = mpatches.Patch(color='blue',  label='Newton (●)')
    patch_olver  = mpatches.Patch(color='red',   label='Olver (▲)')
    ax1.legend(handles=[patch_exact, patch_newton, patch_olver], fontsize=9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(x)')
    ax1.set_title('Polinomul P(x) si radacinile gasite')
    ymax = max(abs(y_plot))
    ax1.set_ylim(-ymax * 0.3, ymax * 0.3)
    ax1.grid(True, alpha=0.3)

    # ─ Graficul 2: Convergenta Newton vs Olver (pas cu pas) ─
    ax2 = axes[1]

    def get_convergenta(metoda_fn, a, x0, eps=EPSILON, kmax=50):
        """Returneaza lista erorilor |xk - x*| la fiecare pas."""
        x_star_approx, _, _ = metoda_fn(a, x0)
        erori = []
        x = x0
        for k in range(1, kmax + 1):
            if metoda_fn == metoda_newton:
                Pv, dPv = horner_P_dP(a, x)
                if abs(dPv) <= eps: break
                delta = Pv / dPv
            else:
                Pv, dPv, d2Pv = horner_P_dP_d2P(a, x)
                if abs(dPv) <= eps: break
                ck = (Pv**2 * d2Pv) / (dPv**3)
                delta = Pv / dPv + 0.5 * ck
            x_nou = x - delta
            erori.append(abs(x_nou - x_star_approx))
            x = x_nou
            if abs(delta) < eps: break
        return erori

    # Convergenta pornind de la x0 aproape de prima radacina
    x0_conv = RADACINI_EXACTE[0] + 0.5
    erori_n = get_convergenta(metoda_newton, a, x0_conv)
    erori_o = get_convergenta(metoda_olver,  a, x0_conv)

    # Filtram zerouri pentru log
    erori_n = [max(e, 1e-16) for e in erori_n]
    erori_o = [max(e, 1e-16) for e in erori_o]

    ax2.semilogy(range(1, len(erori_n) + 1), erori_n, 'b-o', linewidth=2,
                 markersize=6, label=f'Newton ({len(erori_n)} pasi)')
    ax2.semilogy(range(1, len(erori_o) + 1), erori_o, 'r-^', linewidth=2,
                 markersize=6, label=f'Olver ({len(erori_o)} pasi)')
    ax2.axhline(EPSILON, color='gray', linestyle='--', alpha=0.7, label=f'ε = {EPSILON}')
    ax2.set_xlabel('Iteratia k')
    ax2.set_ylabel('|xk - x*|  (scala log)')
    ax2.set_title(f'Convergenta metodelor (x0 = {x0_conv})')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    output_png_path = os.path.join(output_dir, "tema7_grafice.png")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafice salvate: {output_png_path}")

    print(f"\n{separator}")
    print("  GATA! Fisiere generate:")
    print("    - tema7_grafice.png  (grafice)")
    print("    - tema7_radacini.txt (radacinile gasite)")
    print(separator)


if __name__ == "__main__":
    main()
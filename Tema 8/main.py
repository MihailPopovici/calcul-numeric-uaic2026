"""
TEMA 8 - Minimizarea functiilor cu Metoda Gradientului Descendent

Ce face programul:
  - Aproximeaza punctul de minim al unei functii F(x1, x2, ...)
  - Foloseste doua strategii pentru rata de invatare: constanta si backtracking
  - Calculeaza gradientul in doua moduri: analitic (formula exacta) si numeric (aproximativ)
  - Compara toate combinatiile din punctul de vedere al numarului de iteratii
  - Testeaza toate functiile din PDF
  - Face grafice pentru fiecare functie
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  SECTIUNEA 1: DEFINITIILE FUNCTIILOR SI GRADIENTILOR LOR
# ══════════════════════════════════════════════════════════════

"""
Fiecare functie are:
  - F(x)       : valoarea functiei in punctul x = [x1, x2]
  - grad_F(x)  : gradientul analitic (derivatele partiale calculate manual)
  - x_star     : punctul de minim exact (din PDF, pentru verificare)
"""

# ── Functia 1 ──────────────────────────────────────────────
def F1(x):
    """
    F(x1, x2) = x1^2 + x2^2 - 2*x1 - 4*x2 - 1
    Minim la x* = (1, 2)
    """
    x1, x2 = x
    return x1**2 + x2**2 - 2*x1 - 4*x2 - 1

def grad_F1(x):
    """
    Derivata partiala dupa x1: 2*x1 - 2
    Derivata partiala dupa x2: 2*x2 - 4
    """
    x1, x2 = x
    return np.array([2*x1 - 2, 2*x2 - 4])

# ── Functia 2 ──────────────────────────────────────────────
def F2(x):
    """
    F(x1, x2) = 3*x1^2 - 12*x1 + 2*x2^2 + 16*x2 - 10
    Minim la x* = (2, -4)
    """
    x1, x2 = x
    return 3*x1**2 - 12*x1 + 2*x2**2 + 16*x2 - 10

def grad_F2(x):
    x1, x2 = x
    return np.array([6*x1 - 12, 4*x2 + 16])

# ── Functia 3 ──────────────────────────────────────────────
def F3(x):
    """
    F(x1, x2) = x1^2 - 4*x1*x2 + 4.5*x2^2 - 4*x2 + 3
    Minim la x* = (8, 4)
    """
    x1, x2 = x
    return x1**2 - 4*x1*x2 + 4.5*x2**2 - 4*x2 + 3

def grad_F3(x):
    x1, x2 = x
    return np.array([2*x1 - 4*x2, -4*x1 + 9*x2 - 4])

# ── Functia 4 ──────────────────────────────────────────────
def F4(x):
    """
    F(x1, x2) = x1^2*x2 - 2*x1*x2^2 + 3*x1*x2 + 4
    Minim local la x* = (-1, 0.5)
    """
    x1, x2 = x
    return x1**2*x2 - 2*x1*x2**2 + 3*x1*x2 + 4

def grad_F4(x):
    x1, x2 = x
    return np.array([2*x1*x2 - 2*x2**2 + 3*x2,
                     x1**2 - 4*x1*x2 + 3*x1])

# ── Functia 5: loss logistic (din PDF) ────────────────────
def sigma(z):
    """Functia sigmoid: σ(z) = 1 / (1 + e^(-z))"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def F5(x):
    """
    l(w0, w1) = -ln(1 - σ(w0 - w1)) - ln(σ(w0 + w1))
    Functie de pierdere (loss) din machine learning
    """
    w0, w1 = x
    v1 = 1.0 - sigma(w0 - w1)
    v2 = sigma(w0 + w1)
    v1 = max(v1, 1e-15)
    v2 = max(v2, 1e-15)
    return -np.log(v1) - np.log(v2)

def grad_F5(x):
    w0, w1 = x
    s1 = sigma(w0 - w1)
    s2 = sigma(w0 + w1)
    return np.array([s1 + s2 - 1, s2 - s1 - 1])

# ── Lista tuturor functiilor ───────────────────────────────
FUNCTII = [
    {
        "F": F1, "grad": grad_F1,
        "nume": "F1(x) = x1² + x2² - 2x1 - 4x2 - 1",
        "x_star": np.array([1.0, 2.0]),
        "x0": np.array([0.0, 0.0]),
        "domeniu": [(-2, 5), (-2, 7)],
    },
    {
        "F": F2, "grad": grad_F2,
        "nume": "F2(x) = 3x1² - 12x1 + 2x2² + 16x2 - 10",
        "x_star": np.array([2.0, -4.0]),
        "x0": np.array([0.0, 0.0]),
        "domeniu": [(-2, 6), (-8, 2)],
    },
    {
        "F": F3, "grad": grad_F3,
        "nume": "F3(x) = x1² - 4x1x2 + 4.5x2² - 4x2 + 3",
        "x_star": np.array([8.0, 4.0]),
        "x0": np.array([5.0, 2.0]),
        "domeniu": [(0, 15), (0, 8)],
    },
    {
        "F": F4, "grad": grad_F4,
        "nume": "F4(x) = x1²x2 - 2x1x2² + 3x1x2 + 4",
        "x_star": np.array([-1.0, 0.5]),
        "x0": np.array([-0.5, 0.3]),
        "domeniu": [(-3, 2), (-1, 3)],
    },
    {
        "F": F5, "grad": grad_F5,
        "nume": "F5(l) = -ln(1-σ(w0-w1)) - ln(σ(w0+w1))",
        "x_star": None,   # nu avem formula explicita
        "x0": np.array([1.0, 1.0]),
        "domeniu": [(-3, 3), (-3, 3)],
    },
]


# ══════════════════════════════════════════════════════════════
#  SECTIUNEA 2: GRADIENTUL NUMERIC (APROXIMATIV)
# ══════════════════════════════════════════════════════════════

def gradient_numeric(F, x, h=1e-5):
    """
    Aproximeaza gradientul lui F in punctul x folosind formula din PDF:

        dF/dxi ≈ (-F(x+2h*ei) + 8F(x+h*ei) - 8F(x-h*ei) + F(x-2h*ei)) / (12h)

    unde ei = vectorul unitate pe directia i (numai xi e modificat).

    De ce aceasta formula si nu cea simpla (F(x+h)-F(x))/h ?
    Formula din PDF e de ordin 4 in h (eroarea e proportionala cu h^4),
    pe cand formula simpla e de ordin 1. Asadar, cu acelasi h, obtinem
    o aproximare de ~1000x mai precisa.

    Parametri:
      F : functia de minimizat
      x : punctul curent (vector numpy)
      h : pasul de diferentiere (implicit 1e-5)
    """
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        # Construim vectorii cu xi deplasat cu +2h, +h, -h, -2h
        x1 = x.copy(); x1[i] += 2*h   # xi + 2h
        x2 = x.copy(); x2[i] += h     # xi + h
        x3 = x.copy(); x3[i] -= h     # xi - h
        x4 = x.copy(); x4[i] -= 2*h   # xi - 2h

        F1v = F(x1)
        F2v = F(x2)
        F3v = F(x3)
        F4v = F(x4)

        grad[i] = (-F1v + 8*F2v - 8*F3v + F4v) / (12*h)

    return grad


# ══════════════════════════════════════════════════════════════
#  SECTIUNEA 3: RATA DE INVATARE
# ══════════════════════════════════════════════════════════════

def rata_constanta(F, x, grad_x, eta=1e-3):
    """
    Strategia 1: rata de invatare fixa.

    Simpla, dar riscanta:
    - prea mare → sare peste minim, oscileaza sau diverge
    - prea mica  → converge, dar foarte incet (multe iteratii)

    Returneaza: eta (neschimbat)
    """
    return eta


def backtracking(F, x, grad_x, beta=0.8):
    """
    Strategia 2: backtracking line search (ajustare automata a pasului).

    Ideea: incepem cu eta=1 si o micsoram pana cand
    functia 'coboara suficient de mult' in directia gradientului.

    Conditia de coborare (Armijo):
        F(x - eta * grad) <= F(x) - (eta/2) * ||grad||^2

    Daca nu e satisfacuta, inmultim eta cu beta (< 1) si incercam din nou.
    Repetam maxim 8 ori (conform PDF).

    De ce functioneaza? Garanteaza ca facem un pas care chiar scade F,
    adaptat la 'panta locala' a functiei.
    """
    eta = 1.0
    Fx = F(x)
    grad_norm_sq = np.dot(grad_x, grad_x)   # ||grad||^2

    for _ in range(8):
        x_nou = x - eta * grad_x
        if F(x_nou) <= Fx - (eta / 2) * grad_norm_sq:
            break   # conditia satisfacuta, oprim
        eta *= beta  # micsoram pasul

    return eta


# ══════════════════════════════════════════════════════════════
#  SECTIUNEA 4: METODA GRADIENTULUI DESCENDENT
# ══════════════════════════════════════════════════════════════

def gradient_descendent(F, grad_func, x0,
                         metoda_eta="backtracking",
                         eta_const=1e-3,
                         eps=1e-6,
                         kmax=30000):
    """
    Algoritmul principal de minimizare.

    Idee de baza (intuitiv):
      Esti pe un munte si vrei sa ajungi in vale. La fiecare pas,
      te uiti in ce directie coboara cel mai abrupt (= directia -gradient)
      si faci un pas in acea directie. Repeti pana ajungi in vale.

    Formula: x_nou = x_vechi - eta * gradient(x_vechi)

    Parametri:
      F          : functia de minimizat
      grad_func  : functia care calculeaza gradientul
      x0         : punctul de start (ales aleator sau specificat)
      metoda_eta : "constant" sau "backtracking"
      eta_const  : valoarea eta fixa (folosita doar daca metoda="constant")
      eps        : precizia dorita (criteriu de oprire)
      kmax       : numar maxim de iteratii

    Returneaza:
      x          : punctul de minim gasit
      k          : numarul de iteratii efectuate
      converge   : True daca a converges, False daca a divergat
      istoric_x  : lista cu x-urile de la fiecare iteratie (pentru grafic)
      istoric_F  : lista cu F(x) la fiecare iteratie (pentru grafic)
    """
    x = x0.copy().astype(float)
    istoric_x = [x.copy()]
    istoric_F = [F(x)]

    for k in range(1, kmax + 1):
        grad_x = grad_func(x)   # calculam gradientul in x curent

        # Calculam rata de invatare
        if metoda_eta == "constant":
            eta = rata_constanta(F, x, grad_x, eta=eta_const)
        else:
            eta = backtracking(F, x, grad_x)

        # Pasul de actualizare: mergem in directia -gradient
        pas = eta * grad_x

        x = x - pas

        istoric_x.append(x.copy())
        istoric_F.append(F(x))

        # Criteriu de oprire: ||eta * grad|| < eps
        # Adica: pasul facut e suficient de mic
        norma_pas = np.linalg.norm(pas)

        if norma_pas <= eps:
            return x, k, True, istoric_x, istoric_F

        # Protectie la divergenta
        if norma_pas > 1e10 or not np.isfinite(norma_pas):
            return x, k, False, istoric_x, istoric_F

    return x, kmax, False, istoric_x, istoric_F


# ══════════════════════════════════════════════════════════════
#  SECTIUNEA 5: PROGRAMUL PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    eps = 1e-6
    kmax = 30000

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = "=" * 70

    print(separator)
    print("  TEMA 8 – Minimizarea functiilor cu Metoda Gradientului Descendent")
    print(separator)
    print(f"  Precizie eps = {eps},  Iteratii max = {kmax}")

    # Vom face cate un grafic per functie
    rezultate_toate = []

    for idx, func_info in enumerate(FUNCTII):
        F       = func_info["F"]
        grad_an = func_info["grad"]        # gradient analitic
        grad_nu = lambda x, F=F: gradient_numeric(F, x)  # gradient numeric
        x0      = func_info["x0"]
        x_star  = func_info["x_star"]
        nume    = func_info["nume"]

        print(f"\n{separator}")
        print(f"  FUNCTIA {idx+1}: {nume}")
        print(separator)
        if x_star is not None:
            print(f"  Minim exact: x* = {x_star}")
        print(f"  Punct de start: x0 = {x0}")

        # Cele 4 combinatii de testat:
        combinatii = [
            ("Backtracking + Grad.Analitic",  "backtracking", grad_an),
            ("Backtracking + Grad.Numeric",   "backtracking", grad_nu),
            ("Constant + Grad.Analitic",      "constant",     grad_an),
            ("Constant + Grad.Numeric",       "constant",     grad_nu),
        ]

        rezultate_func = []

        print(f"\n  {'Combinatie':<38} | {'Iteratii':>9} | {'F(x*)':>12} | {'Eroare':>12} | {'Status':>10}")
        print(f"  {'-'*38}-+-{'-'*9}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

        for (label, metoda_eta, grad_func) in combinatii:
            x_gasit, k, ok, ist_x, ist_F = gradient_descendent(
                F, grad_func, x0,
                metoda_eta=metoda_eta,
                eta_const=1e-3,
                eps=eps,
                kmax=kmax
            )
            Fx = F(x_gasit)
            eroare = np.linalg.norm(x_gasit - x_star) if x_star is not None else float('nan')
            status = "OK" if ok else "DIVERGE"

            print(f"  {label:<38} | {k:>9} | {Fx:>12.6f} | {eroare:>12.2e} | {status:>10}")

            rezultate_func.append({
                "label": label, "k": k, "ok": ok,
                "ist_x": ist_x, "ist_F": ist_F,
                "x_gasit": x_gasit, "Fx": Fx
            })

        rezultate_toate.append(rezultate_func)

        # ── Grafic pentru aceasta functie ──
        domeniu = func_info["domeniu"]
        x1_range = np.linspace(domeniu[0][0], domeniu[0][1], 300)
        x2_range = np.linspace(domeniu[1][0], domeniu[1][1], 300)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = np.vectorize(lambda a, b: F(np.array([a, b])))(X1, X2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Functia {idx+1}: {nume}", fontsize=11, fontweight='bold')

        # ─ Subplot 1: Harta de contur + traiectoriile ─
        ax = axes[0]
        # Limitam Z pentru vizualizare mai buna (valori extreme strica graficul)
        Z_clip = np.clip(Z, np.nanpercentile(Z, 2), np.nanpercentile(Z, 98))
        contour = ax.contourf(X1, X2, Z_clip, levels=40, cmap='RdYlGn_r', alpha=0.7)
        ax.contour(X1, X2, Z_clip, levels=20, colors='gray', linewidths=0.5, alpha=0.5)
        plt.colorbar(contour, ax=ax, shrink=0.8)

        culori = ['blue', 'red', 'cyan', 'magenta']
        for i_r, rez in enumerate(rezultate_func):
            ist = np.array(rez["ist_x"])
            # Afisam cel mult 200 pasi ca sa nu aglomeream graficul
            pas_afisare = max(1, len(ist) // 200)
            ax.plot(ist[::pas_afisare, 0], ist[::pas_afisare, 1],
                    '-o', color=culori[i_r], markersize=2, linewidth=1.2,
                    alpha=0.8, label=rez["label"][:20])
            # Marcare punct final
            ax.scatter([rez["x_gasit"][0]], [rez["x_gasit"][1]],
                       color=culori[i_r], s=80, marker='*', zorder=6)

        # Punct de start si minim exact
        ax.scatter([x0[0]], [x0[1]], color='white', s=120, marker='o',
                   zorder=7, edgecolors='black', linewidths=2, label='x0 (start)')
        if x_star is not None:
            ax.scatter([x_star[0]], [x_star[1]], color='lime', s=150, marker='*',
                       zorder=8, edgecolors='black', linewidths=1.5, label='x* (minim exact)')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Harta de contur + traiectorii')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

        # ─ Subplot 2: Convergenta F(x) in timp ─
        ax2 = axes[1]
        for i_r, rez in enumerate(rezultate_func):
            ist_F = rez["ist_F"]
            # Normalizat fata de valoarea initiala
            ax2.plot(range(len(ist_F)), ist_F,
                     color=culori[i_r], linewidth=1.5,
                     label=f"{rez['label'][:25]} ({rez['k']} pasi)")

        ax2.set_xlabel('Iteratia k')
        ax2.set_ylabel('F(x^k)')
        ax2.set_title('Scaderea valorii F la fiecare iteratie')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Scala log pe Y daca valorile sunt pozitive
        try:
            if all(v > 0 for v in rezultate_func[0]["ist_F"]):
                ax2.set_yscale('log')
        except:
            pass

        plt.tight_layout()
        plt.savefig(output_dir / f"tema8_functia{idx+1}.png",
                dpi=130, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Grafic salvat: tema8_functia{idx+1}.png")

    # ── Grafic comparativ final: numar iteratii ──
    fig, ax = plt.subplots(figsize=(12, 5))
    labels_combo = ["Back+Analitic", "Back+Numeric", "Const+Analitic", "Const+Numeric"]
    culori_bar = ['steelblue', 'tomato', 'mediumseagreen', 'gold']
    n_func = len(FUNCTII)
    n_combo = len(labels_combo)
    x_pos = np.arange(n_func)
    width = 0.2

    for j in range(n_combo):
        iteratii = []
        for i in range(n_func):
            rez = rezultate_toate[i][j]
            iteratii.append(rez["k"] if rez["ok"] else kmax)
        ax.bar(x_pos + j*width, iteratii, width, label=labels_combo[j],
               color=culori_bar[j], alpha=0.85)

    ax.set_xlabel('Functia')
    ax.set_ylabel('Numar iteratii')
    ax.set_title('Comparatie numar de iteratii: toate combinatiile, toate functiile')
    ax.set_xticks(x_pos + width*1.5)
    ax.set_xticklabels([f"F{i+1}" for i in range(n_func)])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "tema8_comparatie.png", dpi=130, bbox_inches='tight')
    plt.close()

    print(f"\n{separator}")
    print("  GATA! Fisiere generate:")
    for i in range(len(FUNCTII)):
        print(f"    - tema8_functia{i+1}.png")
    print("    - tema8_comparatie.png")
    print(separator)


if __name__ == "__main__":
    main()
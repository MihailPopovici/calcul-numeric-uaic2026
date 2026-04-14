"""
=============================================================================
TEMA 4 - Rezolvarea sistemelor liniare rare cu metoda Gauss-Seidel
=============================================================================

Structura matricei rare A (simetrica):
  - diagonala principala  (d0, n elemente)
  - diagonala secundara de ordin p (d1, n-p elemente): d1[k] = A[k][k+p] = A[k+p][k]
  - diagonala secundara de ordin q (d2, n-q elemente): d2[k] = A[k][k+q] = A[k+q][k]

Relatii indici:
  - Diagonala sup. ordin p: elemntele a_{0,p}, a_{1,p+1}, ..., a_{n-1-p, n-1}
    => a_{i, i+p} = d1[i],  i = 0..n-1-p   (d1 are n-p elemente)
  - Diagonala inf. ordin p: a_{p,0}, a_{p+1,1}, ..., a_{n-1, n-1-p}  (simetrie)
    => a_{i+p, i} = d1[i],  i = 0..n-1-p

Fisierele disponibile (uploadate):
  d0_i.txt  - diagonala principala (n elemente)
  d1_i.txt  - diagonala secundara de ordin p (n-p elemente)
  d2_i.txt  - diagonala secundara de ordin q (n-q elemente)  [nu exista in upload]
  b_i.txt   - termenii liberi (n elemente)                   [nu exista in upload]

Nota: In aceasta implementare, pentru sistemul 1 avem d0_1 si d1_1 disponibile.
      Codul este scris generic pentru orice combinatie de fisiere.
=============================================================================
"""

import numpy as np
import os


# =============================================================================
# PASUL 0: Parametri generali
# =============================================================================

EPSILON = 1e-8       # precizia calculelor (eps = 10^(-p), p=8)
K_MAX   = 10000      # numarul maxim de iteratii Gauss-Seidel
DIV_THR = 1e10       # prag de divergenta


# =============================================================================
# FUNCTIE: incarca_vector
# Citeste un fisier text cu cate un numar real pe linie si returneaza un
# array numpy. Returneaza None daca fisierul nu exista.
# =============================================================================

def incarca_vector(cale_fisier):
    """
    Citeste elementele unui vector dintr-un fisier text (un numar pe linie).
    
    Parametri:
        cale_fisier (str): calea catre fisierul text
        
    Return:
        np.ndarray cu valorile citite, sau None daca fisierul nu exista
    """
    if not os.path.exists(cale_fisier):
        print(f"  [!] Fisierul '{cale_fisier}' nu a fost gasit.")
        return None
    with open(cale_fisier, 'r') as f:
        valori = [float(linie.strip()) for linie in f if linie.strip()]
    return np.array(valori, dtype=np.float64)


# =============================================================================
# FUNCTIE: determina_ordinul_diagonalei
# Din lungimile d0 (n) si d1 (n-p) deducem p = n - len(d1).
# Similar pentru q.
# =============================================================================

def determina_ordinul_diagonalei(n, lungime_d):
    """
    Determina ordinul p (sau q) al diagonalei secundare.
    Daca diagonala de ordin p are n-p elemente, atunci p = n - len(d).
    
    Parametri:
        n (int)         : dimensiunea sistemului
        lungime_d (int) : numarul de elemente din vectorul diagonalei secundare
        
    Return:
        p (int) : ordinul diagonalei
    """
    p = n - lungime_d
    return p


# =============================================================================
# FUNCTIE: verifica_diagonala_principala
# Cerintele spun sa verificam ca |d0[i]| > eps pentru toti i.
# =============================================================================

def verifica_diagonala_principala(d0, eps):
    """
    Verifica ca toate elementele diagonalei principale sunt nenule (|d0[i]| > eps).
    
    Return:
        True  daca toate elementele sunt nenule
        False daca exista cel putin un element nul (metoda nu poate fi aplicata)
    """
    indici_nuli = np.where(np.abs(d0) <= eps)[0]
    if len(indici_nuli) > 0:
        print(f"  [EROARE] Exista {len(indici_nuli)} element(e) nule pe diagonala principala!")
        print(f"           Indici: {indici_nuli[:10]}{'...' if len(indici_nuli)>10 else ''}")
        return False
    print(f"  [OK] Toate cele {len(d0)} elemente ale diagonalei principale sunt nenule.")
    return True


# =============================================================================
# FUNCTIE: gauss_seidel_sparse
#
# Implementarea metodei Gauss-Seidel pentru matricea rara A memorata prin
# vectorii d0, d1, d2 (si opzional mai multi vectori de diagonale).
#
# FORMULA (din enunt):
#   x_c[i] = (b[i] - suma_elemente_nenule_linia_i_fara_diagonal * x[?]) / d0[i]
#
# Elementele nenule de pe linia i (in afara de diagonal) sunt:
#
#   Din d1 (diagonala de ordin p):
#     - elementul de pe coloana i+p: d1[i]     daca i+p <= n-1
#       (element superior: A[i][i+p] = d1[i])
#     - elementul de pe coloana i-p: d1[i-p]   daca i-p >= 0
#       (element inferior: A[i][i-p] = d1[i-p], prin simetrie)
#
#   Din d2 (diagonala de ordin q):
#     - elementul de pe coloana i+q: d2[i]     daca i+q <= n-1
#     - elementul de pe coloana i-q: d2[i-q]   daca i-q >= 0
#
# La Gauss-Seidel:
#   - daca coloana j < i  => folosim x_c[j]  (deja actualizat in iteratia curenta)
#   - daca coloana j > i  => folosim x_p[j]  (din iteratia precedenta)
# =============================================================================

def gauss_seidel_sparse(d0, d1, p, d2, q, b, eps=EPSILON, kmax=K_MAX):
    """
    Rezolva sistemul Ax = b cu metoda Gauss-Seidel folosind memorarea rara.
    
    Parametri:
        d0 (np.ndarray): diagonala principala, n elemente
        d1 (np.ndarray): diagonala secundara de ordin p, (n-p) elemente
                         d1[i] = A[i][i+p] = A[i+p][i]
        p  (int)       : ordinul primei diagonale secundare
        d2 (np.ndarray): diagonala secundara de ordin q, (n-q) elemente
                         d2[i] = A[i][i+q] = A[i+q][i]
                         (poate fi None daca nu exista)
        q  (int)       : ordinul celei de-a doua diagonale secundare (sau None)
        b  (np.ndarray): vectorul termenilor liberi, n elemente
        eps (float)    : precizia de convergenta
        kmax (int)     : numarul maxim de iteratii
        
    Return:
        x_c (np.ndarray) : aproximatia solutiei, sau None daca a divergat
        k   (int)        : numarul de iteratii efectuate
        delta_x (float)  : norma diferentei intre ultimii doi termeni
    """
    n = len(d0)
    
    # Initializam x_c = x_p = 0 (vectorul curent si cel precedent)
    x_c = np.zeros(n, dtype=np.float64)
    x_p = np.zeros(n, dtype=np.float64)
    
    k = 0
    delta_x = np.inf  # diferenta intre iteratii consecutive (norma inf)
    
    while True:
        # Salvam iteratia precedenta
        x_p[:] = x_c
        
        # ----------------------------------------------------------------
        # Calculam noua iteratie x_c component cu component (i = 0..n-1)
        # ----------------------------------------------------------------
        for i in range(n):
            # Incepem cu b[i] si scadem contributiile elementelor nenule de pe linia i
            suma = b[i]
            
            # ---- Contributia din d1 (diagonala de ordin p) ----
            
            # Element SUPERIOR al liniei i: A[i][i+p] = d1[i]
            # (exista doar daca i+p <= n-1, adica i <= n-1-p, adica i < len(d1))
            if i < len(d1):
                coloana = i + p        # coloana elementului nenul
                # coloana > i => folosim valoarea din iteratia precedenta
                suma -= d1[i] * x_p[coloana]
            
            # Element INFERIOR al liniei i: A[i][i-p] = d1[i-p]
            # (exista doar daca i-p >= 0, adica i >= p)
            if i >= p:
                coloana = i - p        # coloana elementului nenul
                # coloana < i => folosim valoarea deja actualizata (x_c)
                suma -= d1[i - p] * x_c[coloana]
            
            # ---- Contributia din d2 (diagonala de ordin q) ----
            if d2 is not None and q is not None:
                
                # Element SUPERIOR al liniei i: A[i][i+q] = d2[i]
                if i < len(d2):
                    coloana = i + q
                    suma -= d2[i] * x_p[coloana]
                
                # Element INFERIOR al liniei i: A[i][i-q] = d2[i-q]
                if i >= q:
                    coloana = i - q
                    suma -= d2[i - q] * x_c[coloana]
            
            # Impartim la elementul diagonal (d0[i] = A[i][i])
            x_c[i] = suma / d0[i]
        
        # Calculam delta_x = ||x_c - x_p||_inf (norma infinit)
        delta_x = np.max(np.abs(x_c - x_p))
        k += 1
        
        # Criterii de oprire:
        if delta_x < eps:
            # Convergenta atinsa!
            break
        if k >= kmax:
            # Numarul maxim de iteratii depasit
            break
        if delta_x > DIV_THR:
            # Divergenta clara
            break
    
    if delta_x < eps:
        return x_c, k, delta_x
    else:
        return None, k, delta_x


# =============================================================================
# FUNCTIE: calculeaza_Ax_sparse
#
# Calculeaza y = A * x folosind memorarea rara, printr-o singura parcurgere
# a vectorilor d0, d1, d2.
#
# Structura:
#   y[i] += d0[i] * x[i]                    (diagonala principala)
#   y[i] += d1[i] * x[i+p]   (i < n-p)      (diagonal sup. ordin p, linia i)
#   y[i+p] += d1[i] * x[i]   (i < n-p)      (diagonal inf. ordin p, linia i+p)
#   Similar pentru d2 si q.
# =============================================================================

def calculeaza_Ax_sparse(d0, d1, p, d2, q, x):
    """
    Calculeaza produsul y = A*x folosind reprezentarea rara a matricei A.
    Parcurge o singura data fiecare vector d0, d1, d2.
    
    Return:
        y (np.ndarray): vectorul rezultat A*x
    """
    n = len(d0)
    y = np.zeros(n, dtype=np.float64)
    
    # ---- Parcurgere d0: contributia diagonalei principale ----
    for i in range(n):
        y[i] += d0[i] * x[i]
    
    # ---- Parcurgere d1: contributia diagonalei de ordin p ----
    # d1[i] = A[i][i+p] = A[i+p][i]
    for i in range(len(d1)):
        y[i]     += d1[i] * x[i + p]   # linia i,    coloana i+p
        y[i + p] += d1[i] * x[i]       # linia i+p,  coloana i  (simetrie)
    
    # ---- Parcurgere d2: contributia diagonalei de ordin q ----
    if d2 is not None and q is not None:
        for i in range(len(d2)):
            y[i]     += d2[i] * x[i + q]
            y[i + q] += d2[i] * x[i]
    
    return y


# =============================================================================
# FUNCTIE PRINCIPALA: rezolva_sistem
# Executa toti pasii ceruti din enunt pentru un sistem dat.
# =============================================================================

def rezolva_sistem(idx, folder=".", eps=EPSILON):
    """
    Rezolva sistemul liniar rar numarul 'idx' (1..5) si afiseaza rezultatele.
    
    Pasi:
      1. Afiseaza dimensiunea sistemului n
      2. Determina si afiseaza ordinele diagonalelor p si q
      3. Verifica ca d0 nu are elemente nule
      4. Aplica metoda Gauss-Seidel
      5. Calculeaza y = A * x_GS
      6. Calculeaza si afiseaza ||A*x_GS - b||_inf
    """
    print("=" * 65)
    print(f"  SISTEM {idx}")
    print("=" * 65)
    
    # ----------------------------------------------------------------
    # Construim caile catre fisiere
    # ----------------------------------------------------------------
    fisier_d0 = os.path.join(folder, f"d0_{idx}.txt")
    fisier_d1 = os.path.join(folder, f"d1_{idx}.txt")
    fisier_d2 = os.path.join(folder, f"d2_{idx}.txt")
    fisier_b  = os.path.join(folder, f"b_{idx}.txt")
    
    # ----------------------------------------------------------------
    # Incarcam vectorii disponibili
    # ----------------------------------------------------------------
    print("\n[1] Incarcare date...")
    d0 = incarca_vector(fisier_d0)
    d1 = incarca_vector(fisier_d1)
    d2 = incarca_vector(fisier_d2)
    b  = incarca_vector(fisier_b)
    
    if d0 is None:
        print("  [!] Fisierul d0 lipseste. Sistem sarit.\n")
        return
    
    # ----------------------------------------------------------------
    # PASUL 1: Dimensiunea sistemului
    # ----------------------------------------------------------------
    n = len(d0)
    print(f"\n[1] Dimensiunea sistemului: n = {n}")
    if b is not None:
        assert len(b) == n, f"Eroare: len(b)={len(b)} != n={n}"
    
    # ----------------------------------------------------------------
    # PASUL 2: Ordinele diagonalelor p si q
    # ----------------------------------------------------------------
    print("\n[2] Determinarea ordinelor diagonalelor secundare:")
    
    p = None
    if d1 is not None:
        p = determina_ordinul_diagonalei(n, len(d1))
        print(f"    d1 are {len(d1)} elemente => ordinul p = n - len(d1) = {n} - {len(d1)} = {p}")
        print(f"    => A[i][i+{p}] = d1[i],  i = 0 .. {len(d1)-1}")
    else:
        print("    d1 nu este disponibil.")
    
    q = None
    if d2 is not None:
        q = determina_ordinul_diagonalei(n, len(d2))
        print(f"    d2 are {len(d2)} elemente => ordinul q = n - len(d2) = {n} - {len(d2)} = {q}")
        print(f"    => A[i][i+{q}] = d2[i],  i = 0 .. {len(d2)-1}")
    else:
        print("    d2 nu este disponibil (sau lipseste fisierul).")
    
    # ----------------------------------------------------------------
    # PASUL 3: Verificarea diagonalei principale
    # ----------------------------------------------------------------
    print("\n[3] Verificarea diagonalei principale (|d0[i]| > eps):")
    diagonala_ok = verifica_diagonala_principala(d0, eps)
    
    if not diagonala_ok:
        print("  Metoda Gauss-Seidel nu poate fi aplicata!\n")
        return
    
    # ----------------------------------------------------------------
    # Verificam ca avem toate datele pentru a rezolva sistemul
    # ----------------------------------------------------------------
    if b is None:
        print("\n[!] Vectorul b lipseste. Nu se poate rezolva sistemul.\n")
        return
    if d1 is None:
        print("\n[!] d1 lipseste. Nu se poate construi matricea completa.\n")
        return
    
    # ----------------------------------------------------------------
    # PASUL 4: Metoda Gauss-Seidel
    # ----------------------------------------------------------------
    print(f"\n[4] Metoda Gauss-Seidel (eps={eps}, kmax={K_MAX}):")
    
    x_gs, k_iter, delta = gauss_seidel_sparse(
        d0, d1, p, d2, q, b, eps=eps, kmax=K_MAX
    )
    
    print(f"    Iteratii efectuate: {k_iter}")
    print(f"    Delta final: ||x_c - x_p||_inf = {delta:.2e}")
    
    if x_gs is None:
        if delta > DIV_THR:
            print("    => DIVERGENTA (delta a depasit pragul 1e10).")
        else:
            print(f"    => Nu a converges in {K_MAX} iteratii.")
        print()
        return
    
    print(f"    => CONVERGENTA atinsa dupa {k_iter} iteratii.")
    print(f"    Primele 10 componente ale solutiei x_GS:")
    for i in range(min(10, len(x_gs))):
        print(f"      x_GS[{i}] = {x_gs[i]:.10f}")
    
    # ----------------------------------------------------------------
    # PASUL 5: Calculul y = A * x_GS (parcurgere unica a d0, d1, d2)
    # ----------------------------------------------------------------
    print("\n[5] Calculul y = A * x_GS (parcurgere unica):")
    y = calculeaza_Ax_sparse(d0, d1, p, d2, q, x_gs)
    print(f"    Primele 10 componente ale lui A*x_GS:")
    for i in range(min(10, len(y))):
        print(f"      y[{i}] = {y[i]:.10f}   (b[{i}] = {b[i]:.10f})")
    
    # ----------------------------------------------------------------
    # PASUL 6: Norma reziduului ||A*x_GS - b||_inf
    # ----------------------------------------------------------------
    print("\n[6] Norma reziduului:")
    reziduu = np.max(np.abs(y - b))
    print(f"    ||A*x_GS - b||_inf = {reziduu:.6e}")
    
    # Evaluare calitate solutie
    if reziduu < eps * 100:
        print(f"    => Solutie BUNA (reziduu mic fata de eps={eps:.0e})")
    else:
        print(f"    => Reziduu MARE - solutia poate fi inexacta")
    
    print()


# =============================================================================
# MAIN - ruleaza pentru toate sistemele disponibile
# =============================================================================

if __name__ == "__main__":
    
    # Calea catre fisierele de date (folderul in care se afla acest script)
    FOLDER = os.path.dirname(os.path.abspath(__file__))
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       TEMA 4 - Sisteme Liniare Rare - Gauss-Seidel          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nPrecizie folosita: eps = {EPSILON:.0e}")
    print(f"Iteratii maxime:   kmax = {K_MAX}")
    print(f"Prag divergenta:   {DIV_THR:.0e}")
    print()
    
    # Incercam toate cele 5 sisteme
    for i in range(1, 6):
        rezolva_sistem(idx=i, folder=FOLDER, eps=EPSILON)
    
    # ----------------------------------------------------------------
    # EXEMPLU MANUAL (din PDF): verificare pe sistemul 5x5 din enunt
    # ----------------------------------------------------------------
    print("=" * 65)
    print("  EXEMPLU VERIFICARE (sistemul 5x5 din enuntul PDF)")
    print("=" * 65)
    print("""
  Matricea A (5x5):
    d0 = [102.5, 104.88, 100.0, 101.3, 102.23]  (diag. principala)
    d1 = [2.5, 1.05, 0.0, 1.0]                  (p=1)
    d2 = [1.1, 0.33]                             (q=3)
    b  = [6.0, 7.0, 8.0, 9.0, 1.0]
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0]  (iteratia initiala din PDF)
    """)
    
    d0_ex = np.array([102.5, 104.88, 100.0, 101.3, 102.23])
    d1_ex = np.array([2.5,   1.05,   0.0,   1.0  ])
    d2_ex = np.array([1.1,   0.33                 ])
    b_ex  = np.array([6.0,   7.0,    8.0,   9.0,  1.0])
    p_ex  = 1
    q_ex  = 3
    
    # Verificare x^(1)_0 manual (din PDF):
    # x1_0 = (6.0 - 2.5*2.0 - 1.1*4.0) / 102.5
    x1_0_manual = (6.0 - 2.5*2.0 - 1.1*4.0) / 102.5
    print(f"  Verificare x^(1)[0] (manual din PDF):")
    print(f"    = (6.0 - 2.5*2.0 - 1.1*4.0) / 102.5 = {x1_0_manual:.8f}")
    
    # Rulam Gauss-Seidel pe exemplul din PDF
    x_gs_ex, k_ex, delta_ex = gauss_seidel_sparse(
        d0_ex, d1_ex, p_ex, d2_ex, q_ex, b_ex, eps=1e-8
    )
    
    if x_gs_ex is not None:
        print(f"\n  Solutia aproximata dupa {k_ex} iteratii:")
        for i in range(5):
            print(f"    x_GS[{i}] = {x_gs_ex[i]:.10f}")
        
        y_ex = calculeaza_Ax_sparse(d0_ex, d1_ex, p_ex, d2_ex, q_ex, x_gs_ex)
        rez_ex = np.max(np.abs(y_ex - b_ex))
        print(f"\n  ||A*x_GS - b||_inf = {rez_ex:.6e}")
    
    print("\n[DONE]")
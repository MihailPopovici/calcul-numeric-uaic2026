"""
=============================================================================
TEMA 5 - Rezolvare completa cu explicatii
=============================================================================

Acest fisier contine rezolvarea completa a Temei 5, care acopera:
  1. Metoda Jacobi  - valori si vectori proprii pentru matrice simetrice (p=n)
  2. Iteratia Cholesky - un sir de matrice ce converge catre o forma diagonala
  3. SVD (Descompunerea dupa valori singulare) - pentru matrice dreptunghiulare (p>n)

Vom folosi exemple concrete din documentul temei ca sa putem verifica rezultatele.
"""

import numpy as np

# ===========================================================================
# SECTIUNEA 0: CONCEPTE INTRODUCTIVE (citeste inainte de cod!)
# ===========================================================================
"""
--- CE SUNT VALORILE SI VECTORII PROPRII? ---

Fie A o matrice patrata n x n.
Un numar λ (lambda) se numeste VALOARE PROPRIE a matricei A daca exista
un vector nenul u (VECTORUL PROPRIU asociat) astfel incat:

        A * u = λ * u

Cu alte cuvinte: cand inmultim matricea A cu vectorul u, obtinem acelasi
vector u inmultit cu un scalar λ. Matricea "scala" vectorul, nu il roteste.

Exemplu simplu: daca A = [[2, 0], [0, 3]], atunci:
  - λ1=2, u1=[1,0]: A*[1,0] = [2,0] = 2*[1,0]  ✓
  - λ2=3, u2=[0,1]: A*[0,1] = [0,3] = 3*[0,1]  ✓

--- DE CE NE INTERESEAZA? ---
Valorile proprii apar in mecanica (frecvente de vibratii), grafica 3D
(PCA/compresie date), Google PageRank, mecanica quantica, si multe altele.

--- MATRICEA SIMETRICA ---
O matrice A este simetrica daca A = A^T (transpusa ei este ea insasi).
Matricele simetrice au MEREU valori proprii REALE (nu complexe) - avantaj!
"""

# ===========================================================================
# SECTIUNEA 1: METODA JACOBI
# ===========================================================================
"""
--- CUM FUNCTIONEAZA METODA JACOBI? ---

Ideea: transformam repetat matricea A prin rotatii, pana devine diagonala.
O matrice diagonala are valorile proprii chiar pe diagonala!

Rotatia Rpq(θ) este o matrice care "anuleaza" elementul (p,q) din A.
La fiecare pas:
  1. Gasim elementul cel mai mare (in valoare absoluta) din afara diagonalei
  2. Calculam unghiul de rotatie θ care il face zero
  3. Aplicam rotatia: A_nou = R * A_vechi * R^T
  4. Acumulam rotatiile in matricea U

La final:
  - A contine valorile proprii pe diagonala
  - U contine vectorii proprii ca coloane
"""

def jacobi_eigenvalues(A_input, epsilon=1e-10, kmax=10000):
    """
    Metoda Jacobi pentru calculul valorilor si vectorilor proprii
    ai unei matrice simetrice reale.

    Parametri:
      A_input : matrice simetrica numpy (n x n)
      epsilon : precizia - algoritmul se opreste cand cel mai mare
                element nediagonal e mai mic decat epsilon
      kmax    : numarul maxim de iteratii

    Returneaza:
      eigenvalues : valorile proprii (diagonala matricei finale)
      U           : matricea vectorilor proprii (coloanele = vectorii proprii)
      k           : numarul de iteratii efectuate
    """
    A = A_input.copy().astype(float)
    n = A.shape[0]
    U = np.eye(n)  # Incepem cu matricea identitate

    for k in range(kmax):
        # --- PASUL 1: Gasim indicii (p, q) ai celui mai mare element
        # nediagonal (cautam doar in triunghiul inferior pentru eficienta,
        # deoarece matricea e simetrica)
        max_val = 0.0
        p, q = 1, 0  # valori initiale
        for i in range(1, n):
            for j in range(i):  # j < i => triunghiul inferior
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j

        # --- CONDITIA DE OPRIRE: daca cel mai mare element nediagonal
        # este deja mai mic decat epsilon, matricea e "suficient de diagonala"
        if max_val < epsilon:
            print(f"  Jacobi converge dupa {k} iteratii (max element off-diag = {max_val:.2e})")
            break

        # --- PASUL 2: Calculam unghiul de rotatie θ
        # Vrem sa anulam elementul A[p,q].
        # Formula: alpha = cot(2θ) = (A[p,p] - A[q,q]) / (2 * A[p,q])
        #
        # Alegem t = tan(θ) cu |t| minim (adica θ in [-π/4, π/4])
        # pentru a minimiza perturbatiile la fiecare pas.

        if A[p, q] == 0:  # deja zero, sarind
            continue

        alpha = (A[p, p] - A[q, q]) / (2.0 * A[p, q])

        # Formula pentru t = tan(θ):
        # t = -alpha + sign(alpha) * sqrt(alpha^2 + 1)
        # Daca alpha = 0, t = 1 (θ = 45 grade)
        if alpha >= 0:
            t = -alpha + np.sqrt(alpha**2 + 1)
        else:
            t = -alpha - np.sqrt(alpha**2 + 1)

        # c = cos(θ), s = sin(θ)
        c = 1.0 / np.sqrt(1 + t**2)
        s = t / np.sqrt(1 + t**2)

        # --- PASUL 3: Actualizam matricea A "in-place" (fara matrice auxiliara)
        # Actualizam toate elementele din liniile/coloanele p si q

        # Salvam valorile vechi pentru coloanele p si q (avem nevoie mai jos)
        old_ap = A[p, :].copy()
        old_aq = A[q, :].copy()

        # Actualizam elementele care nu sunt pe pozitiile (p,p), (q,q), (p,q):
        for j in range(n):
            if j != p and j != q:
                A[p, j] = c * old_ap[j] + s * old_aq[j]
                A[j, p] = A[p, j]  # simetrie

                A[q, j] = -s * old_ap[j] + c * old_aq[j]
                A[j, q] = A[q, j]  # simetrie

        # Actualizam elementele diagonale (pp) si (qq):
        # Folosim formula simplificata: b_pp = a_pp + t * a_pq
        A[p, p] = old_ap[p] + t * old_ap[q]
        A[q, q] = old_aq[q] - t * old_ap[q]

        # Elementul (p,q) devine zero (scopul rotatiei):
        A[p, q] = 0.0
        A[q, p] = 0.0

        # --- PASUL 4: Actualizam matricea U a vectorilor proprii
        # V = U * R^T(θ)  =>  modificam coloanele p si q ale lui U
        old_up = U[:, p].copy()
        old_uq = U[:, q].copy()

        U[:, p] = c * old_up + s * old_uq
        U[:, q] = -s * old_up + c * old_uq

    else:
        print(f"  ATENTIE: Jacobi nu a convergit in {kmax} iteratii!")

    # Valorile proprii sunt pe diagonala matricei A (dupa transformari)
    eigenvalues = np.diag(A)
    return eigenvalues, U, k


def demo_jacobi():
    print("=" * 70)
    print("SECTIUNEA 1: METODA JACOBI - Valori si Vectori Proprii")
    print("=" * 70)

    print("""
--- TEORIA PE SCURT ---
Metoda Jacobi transforma repetat matricea A prin rotatii ortogonale.
La final, A devine (aproximativ) diagonala, cu valorile proprii pe diagonala,
iar coloanele matricei U acumulate contin vectorii proprii.

Verificam: A_init * U ≈ U * Λ
unde Λ = diag(λ1, λ2, ..., λn) este matricea diagonala a valorilor proprii.
Norma ||A_init * U - U * Λ|| trebuie sa fie aproape de zero.
""")

    # --- Exemplul 1 din tema ---
    print("--- EXEMPLUL 1 (din tema): ---")
    A1 = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    print("Matricea A =")
    print(A1)
    print("Valori proprii asteptate: λ1=-1, λ2=0, λ3=2\n")

    A_init1 = A1.copy()
    eigenvalues1, U1, iters1 = jacobi_eigenvalues(A1)

    # Sortam valorile proprii pentru comparatie
    idx = np.argsort(eigenvalues1)
    eigenvalues1_sorted = eigenvalues1[idx]
    U1_sorted = U1[:, idx]

    print(f"Valori proprii calculate: {eigenvalues1_sorted}")
    print(f"Valori proprii asteptate: [-1,  0,  2]")
    print()
    print("Vectori proprii (coloanele lui U):")
    for i in range(len(eigenvalues1_sorted)):
        print(f"  λ={eigenvalues1_sorted[i]:.6f}  →  u = {U1_sorted[:, i]}")

    # Verificare: A_init * U ≈ U * Λ
    Lambda1 = np.diag(eigenvalues1)   # matricea Λ (nevalorata)
    U1_orig = U1                       # U nesortat, pentru verificare exacta
    diff1 = A_init1 @ U1_orig - U1_orig @ Lambda1
    print(f"\nVerificare: ||A_init * U - U * Λ|| = {np.linalg.norm(diff1):.2e}")
    print("(Trebuie sa fie aproape de zero - confirma ca am calculat corect!)")

    # --- Exemplul 2 din tema ---
    print("\n--- EXEMPLUL 2 (din tema): ---")
    A2 = np.array([
        [1.0, 1.0, 2.0],
        [1.0, 1.0, 2.0],
        [2.0, 2.0, 2.0]
    ])
    print("Matricea A =")
    print(A2)
    ev2_asteptate = [0, 2*(1-np.sqrt(2)), 2*(1+np.sqrt(2))]
    print(f"Valori proprii asteptate: {[round(v,6) for v in ev2_asteptate]}\n")

    A_init2 = A2.copy()
    eigenvalues2, U2, iters2 = jacobi_eigenvalues(A2)

    idx2 = np.argsort(eigenvalues2)
    ev2_sorted = eigenvalues2[idx2]
    U2_sorted = U2[:, idx2]

    print(f"Valori proprii calculate (sortate): {np.round(ev2_sorted, 6)}")

    Lambda2 = np.diag(eigenvalues2)
    diff2 = A_init2 @ U2 - U2 @ Lambda2
    print(f"Verificare: ||A_init * U - U * Λ|| = {np.linalg.norm(diff2):.2e}")

    # --- Matrice generica ---
    print("\n--- EXEMPLU GENERIC (matrice simetrica 4x4 aleatoare): ---")
    np.random.seed(42)
    B = np.random.randint(1, 10, (4, 4)).astype(float)
    A_gen = B + B.T  # O facem simetrica adaugand transpusa
    print("Matricea A (simetrica 4x4) =")
    print(A_gen)

    A_init_gen = A_gen.copy()
    ev_gen, U_gen, iters_gen = jacobi_eigenvalues(A_gen)

    idx_gen = np.argsort(ev_gen)
    ev_gen_sorted = ev_gen[idx_gen]
    print(f"\nValori proprii (sortate): {np.round(ev_gen_sorted, 6)}")
    print("Verificare cu numpy.linalg.eigh (referinta):")
    ev_numpy, _ = np.linalg.eigh(A_init_gen)
    print(f"Valori proprii numpy (sortate): {np.round(ev_numpy, 6)}")

    Lambda_gen = np.diag(ev_gen)
    diff_gen = A_init_gen @ U_gen - U_gen @ Lambda_gen
    print(f"\nVerificare finala: ||A_init * U - U * Λ|| = {np.linalg.norm(diff_gen):.2e}")
    print("(Valoare mica => metoda functioneaza corect!)")


# ===========================================================================
# SECTIUNEA 2: ITERATIA CHOLESKY
# ===========================================================================
"""
--- CE ESTE DESCOMPUNEREA CHOLESKY? ---

Orice matrice simetrica pozitiv definita A poate fi scrisa ca:
        A = L * L^T
unde L este o matrice inferior triunghiulara (cu elemente nenule doar pe
diagonala si sub ea). Aceasta se numeste DESCOMPUNEREA CHOLESKY.

--- CE FACE ITERATIA CHOLESKY? ---

Construim un sir de matrice:
  A(0) = A
  A(1) = L0^T * L0    (inversam ordinea!)
  A(2) = L1^T * L1
  ...

Acest sir CONVERGE catre o matrice DIAGONALA cu valorile proprii pe diagonala!
Este o alternativa la metoda QR clasica, valabila pentru matrice pozitiv definite.

NOTA: Pentru ca aceasta iteratie sa functioneze, A trebuie sa fie
pozitiv definita (toate valorile proprii > 0). Daca nu e, adaugam un
multiplu al matricei identitate: A + c*I, cu c ales corespunzator.
"""

def iteratie_cholesky(A_input, epsilon=1e-10, kmax=1000):
    """
    Calculam sirul de matrice prin iteratia Cholesky.

    La fiecare pas:
      1. Descompunem A(k) = L * L^T  (Cholesky)
      2. Calculam A(k+1) = L^T * L  (inversam ordinea)
      3. Verificam daca ||A(k+1) - A(k)|| < epsilon

    Parametri:
      A_input : matrice simetrica pozitiv definita (n x n)
      epsilon : precizia pentru criteriul de convergenta
      kmax    : numarul maxim de iteratii

    Returneaza:
      A_final : ultima matrice calculata
      k       : numarul de iteratii
    """
    A = A_input.copy().astype(float)
    n = A.shape[0]

    print(f"\n  Incepem iteratia Cholesky (epsilon={epsilon}, kmax={kmax})")
    print(f"  Matricea initiala A(0):\n{np.round(A, 4)}\n")

    for k in range(kmax):
        A_prev = A.copy()

        # Pasul 1: Descompunere Cholesky A = L * L^T
        # numpy.linalg.cholesky(A) returneaza L (inferior triunghiulara)
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            print(f"  ATENTIE: Matricea nu mai e pozitiv definita la iteratia {k}!")
            print("  Aceasta poate aparea din cauza erorilor de rotunjire.")
            break

        # Pasul 2: A(k+1) = L^T * L
        A = L.T @ L

        # Pasul 3: Verificam convergenta
        diff_norm = np.linalg.norm(A - A_prev)

        if k < 5 or k % 100 == 0:  # Afisam primele 5 iteratii + la fiecare 100
            print(f"  Iter {k+1:4d}: ||A(k) - A(k-1)|| = {diff_norm:.2e}")

        if diff_norm < epsilon:
            print(f"\n  Convergenta atinsa dupa {k+1} iteratii!")
            print(f"  Ultima diferenta: {diff_norm:.2e} < epsilon={epsilon}")
            break
    else:
        print(f"\n  S-au atins {kmax} iteratii fara convergenta completa.")

    return A, k + 1


def demo_cholesky():
    print("\n" + "=" * 70)
    print("SECTIUNEA 2: ITERATIA CHOLESKY")
    print("=" * 70)

    print("""
--- TEORIA PE SCURT ---
Construim sirul: A(0) = A, A(1) = L0^T * L0, A(2) = L1^T * L1, ...
Sirul converge la o matrice DIAGONALA cu valorile proprii pe diagonala.

CE FORMA ARE MATRICEA FINALA?
  - (Aproximativ) diagonala!
  - Pe diagonala se gasesc valorile proprii ale matricei initiale.
  - Elementele de sub/deasupra diagonalei tind spre zero.

CE INFORMATII CONTINE?
  - Valorile proprii ale matricei originale (pe diagonala)
  - In general, sunt ordonate DESCRESCATOR (cea mai mare prima)
""")

    # --- Exemplu 1: Matrice 3x3 simetrica pozitiv definita ---
    print("--- EXEMPLU 1: Matrice 3x3 ---")
    # Cream o matrice simetrica pozitiv definita: B^T * B + I garanteaza asta
    np.random.seed(7)
    B = np.random.randn(6, 6)
    A1 = B.T @ B + 6 * np.eye(6)  # pozitiv definita sigur
    #A1 = (A1 + A1.T) / 2  # asiguram simetria exacta

    print("Matricea A (simetrica, pozitiv definita):")
    print(np.round(A1, 4))

    # Valorile proprii reale (referinta)
    ev_ref = np.sort(np.linalg.eigvalsh(A1))[::-1]
    print(f"\nValori proprii reale (referinta, descrescator): {np.round(ev_ref, 6)}")

    A_final1, iters1 = iteratie_cholesky(A1, epsilon=1e-8, kmax=500)

    print(f"\nMatricea finala A({iters1}) :")
    print(np.round(A_final1, 6))

    diag_final = np.sort(np.diag(A_final1))[::-1]
    print(f"\nDiagonala matricei finale (sortata desc): {np.round(diag_final, 6)}")
    print(f"Valori proprii reale (referinta):         {np.round(ev_ref, 6)}")
    print(f"\n=> Diagonala ≈ Valori proprii: diferenta = {np.max(np.abs(diag_final - ev_ref)):.2e}")

    off_diag_norm = np.linalg.norm(A_final1 - np.diag(np.diag(A_final1)))
    print(f"Norma elementelor nediagonale: {off_diag_norm:.2e} (trebuie sa fie aproape de 0)")

    # --- Exemplu 2: Matrice din tema ---
    print("\n--- EXEMPLU 2: Matrice 4x4 din tema ---")
    A2 = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0]
    ])
    print("Matricea A:")
    print(A2)

    # Verificam daca e pozitiv definita
    ev_check = np.linalg.eigvalsh(A2)
    print(f"Valorile proprii: {np.round(ev_check, 4)}")
    print("Aceasta matrice NU e pozitiv definita (are valori proprii negative/zero).")
    print("Adaugam c*I pentru a o face pozitiv definita...")

    # Adaugam un multiplu al identitatii ca sa facem matricea pozitiv definita
    min_ev = min(ev_check)
    c = abs(min_ev) + 1.0  # suficient de mare
    A2_pd = A2 + c * np.eye(4)
    print(f"A_pd = A + {c:.1f}*I")
    print(f"Noile valori proprii: {np.round(np.linalg.eigvalsh(A2_pd), 4)} (toate > 0 ✓)")

    ev_ref2 = np.sort(np.linalg.eigvalsh(A2_pd))[::-1]
    A_final2, iters2 = iteratie_cholesky(A2_pd, epsilon=1e-8, kmax=1000)

    print(f"\nMatricea finala A({iters2}):")
    print(np.round(A_final2, 6))

    diag2 = np.sort(np.diag(A_final2))[::-1]
    print(f"\nDiagonala matricei finale: {np.round(diag2, 6)}")
    print(f"Valori proprii referinta:  {np.round(ev_ref2, 6)}")


# ===========================================================================
# SECTIUNEA 3: SVD - DESCOMPUNEREA DUPA VALORI SINGULARE (p > n)
# ===========================================================================
"""
--- CE ESTE SVD? ---

Orice matrice A (de orice dimensiune p x n) poate fi scrisa ca:
        A = U * S * V^T

unde:
  U  (p x p): matrice ortogonala (coloanele sunt "vectorii singulari stangi")
  S  (p x n): matrice cu valori singulare σ1 >= σ2 >= ... >= 0 pe diagonala
  V  (n x n): matrice ortogonala (coloanele sunt "vectorii singulari drepti")

--- CE SUNT VALORILE SINGULARE? ---
Valorile singulare σi sunt radacinile patrate ale valorilor proprii
ale matricei A^T * A. Sunt intotdeauna >= 0.

--- CE PUTEM CALCULA DIN SVD? ---
1. RANGUL: numarul de valori singulare strict pozitive
2. NUMARUL DE CONDITIONARE: σmax / σmin (masura stabilitatii numerice)
3. PSEUDOINVERSA MOORE-PENROSE: A^I = V * S^I * U^T
   unde S^I inlocuieste fiecare σi > 0 cu 1/σi

--- PSEUDOINVERSA - DE CE? ---
Cand sistemul Ax = b nu are solutie exacta (p > n => sistem supradeterminat),
pseudoinversa da solutia in sensul celor mai mici patrate:
        x^I = A^I * b  minimizeaza ||Ax - b||^2
"""

def pseudoinversa_SI(S_matrix, p, n, sigma_threshold=1e-10):
    """
    Calculeaza matricea S^I (n x p) din S (p x n).

    S^I inverseaza valorile singulare nenule:
      - σi > threshold => S^I[i,i] = 1/σi
      - σi ≈ 0         => S^I[i,i] = 0  (nu impartim la zero!)
    """
    SI = np.zeros((n, p))
    min_dim = min(p, n)
    for i in range(min_dim):
        sigma = S_matrix[i, i] if i < S_matrix.shape[0] and i < S_matrix.shape[1] else 0
        if sigma > sigma_threshold:
            SI[i, i] = 1.0 / sigma
    return SI


def demo_svd():
    print("\n" + "=" * 70)
    print("SECTIUNEA 3: SVD (Descompunerea dupa Valori Singulare) - p > n")
    print("=" * 70)

    print("""
--- TEORIA PE SCURT ---
Pentru o matrice A (p x n) cu p > n (mai multe linii decat coloane):
  A = U * S * V^T
  
  U (p x p): matrice ortogonala
  S (p x n): valori singulare pe diagonala, rest zerouri
  V (n x n): matrice ortogonala

Calculam:
  1. Valorile singulare  => diagonala lui S
  2. Rangul              => numarul de valori singulare > 0
  3. Numarul de conditionare => σmax / σmin
  4. Pseudoinversa A^I = V * S^I * U^T
  5. Pseudoinversa alternativa A^J = (A^T * A)^-1 * A^T
  6. ||A^I - A^J||_1 (trebuie sa fie aproape de zero!)
""")

    # --- Exemplu 1: Matrice 4x3 (p=4 > n=3) ---
    print("--- EXEMPLU 1: Matrice A (4 x 3) ---")
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    print("Matricea A (p=4, n=3):")
    print(A)

    p, n = A.shape  # p=4, n=3

    # -----------------------------------------------------------------------
    # 1. CALCULAM SVD
    # numpy.linalg.svd(A, full_matrices=True) returneaza U, s, Vt
    # unde s e vectorul valorilor singulare (nu matricea S!), Vt = V^T
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T  # V = (Vt)^T

    # Reconstruim matricea S (p x n) din vectorul s
    S_matrix = np.zeros((p, n))
    for i in range(min(p, n)):
        S_matrix[i, i] = s[i]

    print(f"\n1. VALORILE SINGULARE ale lui A:")
    print(f"   σ = {np.round(s, 6)}")
    print(f"   (sunt pe diagonala matricei S, in ordine descrescatoare)")

    # -----------------------------------------------------------------------
    # 2. RANGUL
    threshold_rang = 1e-10
    rang = np.sum(s > threshold_rang)
    print(f"\n2. RANGUL matricei A = {rang}")
    print(f"   (numarul de valori singulare strict pozitive: {np.sum(s > threshold_rang)} din {len(s)})")
    print(f"   Verif. cu numpy: rang = {np.linalg.matrix_rank(A)}")

    # -----------------------------------------------------------------------
    # 3. NUMARUL DE CONDITIONARE
    s_pozitive = s[s > threshold_rang]
    sigma_max = s_pozitive[0]   # cel mai mare (s e sortat descrescator)
    sigma_min = s_pozitive[-1]  # cel mai mic strict pozitiv
    cond_manual = sigma_max / sigma_min

    print(f"\n3. NUMARUL DE CONDITIONARE:")
    print(f"   k2(A) = σmax / σmin = {sigma_max:.4f} / {sigma_min:.4f} = {cond_manual:.4f}")
    print(f"   Verif. cu numpy: k2(A) = {np.linalg.cond(A):.4f}")
    print(f"   (Un numar mare => matrice 'rau conditionata' => calcule mai putin stabile)")

    # -----------------------------------------------------------------------
    # 4. PSEUDOINVERSA MOORE-PENROSE: A^I = V * S^I * U^T
    SI = pseudoinversa_SI(S_matrix, p, n)
    AI = V @ SI @ U.T  # Pseudoinversa Moore-Penrose

    print(f"\n4. PSEUDOINVERSA MOORE-PENROSE A^I (= V * S^I * U^T), dimensiune {AI.shape}:")
    print(np.round(AI, 6))
    print(f"   Verif. cu numpy.linalg.pinv: norma diferenta = {np.linalg.norm(AI - np.linalg.pinv(A)):.2e}")

    # -----------------------------------------------------------------------
    # 5. PSEUDOINVERSA ALTERNATIVA: A^J = (A^T * A)^-1 * A^T
    # Aceasta formula functioneaza NUMAI daca A^T * A este inversabila,
    # adica daca rang(A) = n (coloane liniar independente)
    AtA = A.T @ A  # Matrice n x n
    print(f"\n5. PSEUDOINVERSA A^J = (A^T * A)^-1 * A^T:")

    if np.linalg.matrix_rank(AtA) == n:
        AJ = np.linalg.inv(AtA) @ A.T
        print(np.round(AJ, 6))

        # -----------------------------------------------------------------------
        # 6. NORMA DIFERENTEI
        diff_norm1 = np.linalg.norm(AI - AJ, ord=1)  # norma 1
        print(f"\n6. ||A^I - A^J||_1 = {diff_norm1:.2e}")
        print(f"   (Trebuie sa fie APROAPE DE ZERO - confirma echivalenta celor doua formule!)")
    else:
        print("   ATENTIE: A^T * A nu este inversabila (coloane liniar dependente)!")
        print("   A^J nu se poate calcula prin aceasta formula.")
        AJ = None

    # -----------------------------------------------------------------------
    # --- Exemplu 2: Matrice mai interesanta (rang complet) ---
    print("\n--- EXEMPLU 2: Matrice A (5 x 3) cu rang complet ---")
    np.random.seed(42)
    A2 = np.random.randn(5, 3)  # Matrice aleatoare 5x3 (probabil rang complet)
    p2, n2 = A2.shape
    print(f"Matricea A ({p2} x {n2}):")
    print(np.round(A2, 4))

    U2, s2, Vt2 = np.linalg.svd(A2, full_matrices=True)
    V2 = Vt2.T
    S2 = np.zeros((p2, n2))
    for i in range(min(p2, n2)):
        S2[i, i] = s2[i]

    print(f"\nValori singulare: {np.round(s2, 6)}")
    print(f"Rangul: {np.sum(s2 > 1e-10)}")
    print(f"Numarul de conditionare: {s2[0]/s2[-1]:.4f}")

    SI2 = pseudoinversa_SI(S2, p2, n2)
    AI2 = V2 @ SI2 @ U2.T
    AJ2 = np.linalg.inv(A2.T @ A2) @ A2.T

    diff2 = np.linalg.norm(AI2 - AJ2, ord=1)
    print(f"\nPseudoinversa Moore-Penrose A^I ({AI2.shape[0]} x {AI2.shape[1]}):")
    print(np.round(AI2, 6))
    print(f"\n||A^I - A^J||_1 = {diff2:.2e}  (aproape de zero => formule echivalente ✓)")

    # Demonstratie: A^I * A ≈ I_n (proprietate a pseudoinversei)
    print(f"\nBonus - Verificare proprietate: A^I * A ≈ I_{n2}:")
    print(np.round(AI2 @ A2, 4))
    print("(Trebuie sa fie aproape de matricea identitate!)")


# ===========================================================================
# MAIN - Rulam toate sectiunile
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEMA 5 - REZOLVARE COMPLETA")
    print("Metoda Jacobi | Iteratia Cholesky | SVD")
    print("=" * 70 + "\n")

    # Sectiunea 1: Metoda Jacobi (p = n, matrice simetrica)
    demo_jacobi()

    # Sectiunea 2: Iteratia Cholesky (p = n, matrice simetrica pozitiv definita)
    demo_cholesky()

    # Sectiunea 3: SVD (p > n, matrice dreptunghiulara)
    demo_svd()

    print("\n" + "=" * 70)
    print("REZUMAT FINAL")
    print("=" * 70)
    print("""
1. METODA JACOBI (p=n, A simetrica):
   - Rotatii repetate transforma A catre o matrice diagonala
   - Diagonala finala = valorile proprii
   - Coloanele lui U = vectorii proprii
   - Verificare: ||A_init * U - U * Λ|| ≈ 0

2. ITERATIA CHOLESKY (p=n, A simetrica pozitiv definita):
   - Sir: A(0) = A, A(k+1) = L_k^T * L_k  (unde A(k) = L_k * L_k^T)
   - Converge catre o matrice DIAGONALA
   - Diagonala finala contine valorile proprii ale lui A
   - Necesita matrice pozitiv definita (se poate forta adaugand c*I)

3. SVD (p>n, matrice dreptunghiulara):
   - A = U * S * V^T
   - Rang(A) = nr. valori singulare strict pozitive
   - k2(A) = σmax / σmin (numarul de conditionare)
   - A^I = V * S^I * U^T  (pseudoinversa Moore-Penrose)
   - A^J = (A^T*A)^-1 * A^T  (pseudoinversa in sensul celor mai mici patrate)
   - ||A^I - A^J||_1 ≈ 0  (cele doua formule dau acelasi rezultat)
""")
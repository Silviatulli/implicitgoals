# La politique de requêtes « Ratio »
## 1. Notations

| symbole | sens |
|---|---|
| B | les bottlenecks qu'on peut demander, n = \|B\| ; s désigne un bottleneck |
| Φ = {ϕ_1..ϕ_m} | les hypothèses (Algorithme 1) : sous-ensembles maximaux de B que le robot peut visiter ensemble |
| I_G | les implicit subgoals de l'humain évalué, inconnus du robot |
| K_I | bottlenecks demandés et répondus OUI (K_I ⊆ I_G) |
| K_¬ | bottlenecks demandés et répondus NON (`K_not` dans le code) |
| p_s | a priori P(OUI \| s) |
| T_k = B ∖ ϕ_k | le **certificat** de l'hypothèse k |
| alive = {k : K_I ⊆ ϕ_k} | hypothèses qui contiennent encore tous les OUI |
| U_k = T_k ∖ K_¬ | ce qu'il reste à obtenir en NON pour certifier k ; r_k = \|U_k\| |

## 2. Sur quoi elle se fonde

### 2.1 Certifier, pas identifier

`_terminal` arrête l'épisode :
- en **échec** quand alive = ∅ ;
- en **succès** quand B ∖ K_¬ tient dans une hypothèse, c'est-à-dire quand
  T_k ⊆ K_¬ pour un k de alive.

Le robot n'a donc pas besoin d'identifier I_G ni « la » bonne hypothèse. Il lui
suffit de **prouver** qu'une hypothèse ϕ_k contient I_G, en obtenant un NON sur
chaque bottleneck de T_k. Plusieurs hypothèses contiennent souvent I_G, et les
départager est du travail perdu. C'est le défaut de fond de H1 Info Gain et de
H3 Query Frequency.

Deux conséquences :
- **un NON n'élimine aucune hypothèse** (seul un OUI hors de ϕ_k tue k) ; un
  NON fait seulement avancer les certificats. log2|Φ| n'est donc pas une borne
  du nombre de requêtes, ni par en haut ni par en bas ;
- la borne pertinente est la **borne clairvoyante**
  L*(I_G) = min_{k : I_G ⊆ ϕ_k} |T_k| (et 1 si I_G est incompatible, dans nos
  données). Aucune politique ne fait mieux, épisode par épisode.

### 2.2 Un épisode évalue un DNF monotone

Avec y_s = 1 si la réponse à s est NON :

    succès  ⇔  f(y) = OR_k  AND_{s ∈ T_k}  y_s  = 1

Un épisode est l'**évaluation stochastique d'une fonction booléenne** (SBFE,
*sequential testing*) : chaque requête coûte 1, les réponses sont aléatoires
selon un a priori, et on veut connaître f avec le moins de requêtes possible.
Succès = un terme entièrement à 1 (1-certificat) ; échec = un 0 dans chaque
terme (0-certificat).

### 2.3 La règle par ratio (Smith)

Pour un **seul AND** de tests indépendants, poser d'abord le test le plus
susceptible d'échouer (*fail-fast*) est optimal. Le coût attendu d'un terme k
évalué ainsi, avec U_k trié par p décroissant (q_1 ≥ q_2 ≥ …), et sa
probabilité de succès sous l'a priori produit sont :

    E_k = Σ_{i=1..r_k} Π_{j<i} (1 − q_j)        P_k = Π_i (1 − q_i)

Pour un **OR de ANDs disjoints**, traiter les termes un par un par
ρ_k = P_k / E_k décroissant est optimal. L'argument d'échange est court :
évaluer k puis l coûte en moyenne E_k + (1 − P_k)·E_l, et l puis k coûte
E_l + (1 − P_l)·E_k ; k passe d'abord ssi P_k/E_k ≥ P_l/E_l. ρ_k se lit comme
une **chance de succès par requête dépensée**.

## 3. Ce qu'elle fait concrètement

**L'idée.** Avant chaque question, le robot se demande : « parmi les hypothèses
encore possibles, laquelle suis-je le plus près de certifier, compte tenu de ce
qu'il lui manque et du risque que l'humain réponde OUI ? » Il en fait sa
**cible**. Puis, parmi les NON qui manquent à cette cible, il demande celui qui
sert au plus grand nombre d'autres hypothèses intéressantes. Il recommence
après chaque réponse.

Cela se fait en cinq étapes, qui sont exactement les lignes de
[`GreedyQNet._ratio_shared`](bottlenecks.py#L1282).

> **Deux remarques pour lire le code.**
> - `self.T` n'est **pas** le certificat $T_k$ : c'est la matrice des
>   hypothèses (`_hypothesis_matrix`), avec `self.T[k, s]` vrai ssi
>   $s \in \phi_k$. Le certificat $T_k = B \setminus \phi_k$ s'écrit donc
>   `~self.T[k]`.
> - `KI` et `KN` sont les masques booléens de $K_I$ et $K_\neg$, de forme
>   `(b, n)`. Le code traite `b` états de connaissance à la fois ; on peut
>   oublier cette première dimension en lisant.

### Étape 1 — Quelles hypothèses sont encore en jeu ?

$$
\text{alive} = \{\, k : K_I \subseteq \phi_k \,\}
$$

Une hypothèse ne meurt que si l'humain a dit OUI à un bottleneck qu'elle ne
contient pas. Un NON ne tue jamais rien : l'humain n'a pas besoin de tout
$\phi_k$, seulement que $I_G \subseteq \phi_k$.

Code : `C = self._consistent(KI)`, de forme `(b, m)`.

### Étape 2 — Que manque-t-il à chacune ?

$$
U_k = (B \setminus \phi_k) \setminus K_\neg = T_k \setminus K_\neg ,
\qquad r_k = |U_k|
$$

Ce sont les bottlenecks hors de $\phi_k$ auxquels l'humain n'a pas encore
répondu NON. Quand $U_k$ est vide, $\phi_k$ est certifiée et l'épisode réussit.

Code : `U = C[:, :, None] & ~self.T[None] & ~KN[:, None, :]`, de forme
`(b, m, n)`. Le facteur `C` met à vide les $U_k$ des hypothèses mortes.

### Étape 3 — Combien rapporte chacune ?

On note $q_1, \dots, q_{r_k}$ les $p_s$ des bottlenecks de $U_k$.
- Si on a un prior $p \ne \tfrac12$ (non homogène), on range $U_k$ du
  bottleneck le plus risqué (le plus susceptible d'un OUI) au moins risqué :
  $q_1 \ge q_2 \ge \dots \ge q_{r_k}$.
- Sinon, tous les $q_i$ valent $\tfrac12$ et l'ordre n'a aucune importance
  (c'est le cas du pipeline, voir plus bas).

Alors

$$
P_k = \prod_{i=1}^{r_k} (1 - q_i),
\qquad
E_k = \sum_{i=1}^{r_k} \prod_{j<i} (1 - q_j),
\qquad
\rho_k = \frac{P_k}{E_k}.
$$

- $P_k$ est la probabilité que l'humain réponde NON à tout $U_k$, donc que
  $\phi_k$ finisse certifiée.
- $E_k$ est le nombre moyen de questions pour trancher sur $\phi_k$ si on pose
  d'abord les plus risquées et qu'on s'arrête au premier OUI. La $i$-ème
  question n'est posée que si les $i-1$ précédentes ont donné NON, d'où le
  produit.
- $\rho_k$ est donc une **chance de succès par question dépensée** : une
  hypothèse rapporte si elle est probable *et* courte à finir.

Code :

| ligne | ce qu'elle calcule |
|---|---|
| `q = -np.sort(-np.where(U, p, -1.0), axis=2)` | les $q_i$ de chaque $U_k$, triés par $p$ décroissant ; les bits hors de $U_k$ valent $-1$ et passent à la fin |
| `valid = q >= 0` | vrai pour les $r_k$ premières cases, les vrais $q_i$ |
| `surv = np.cumprod(...)` | `surv[i]` $= \prod_{j \le i} (1 - q_j)$, la probabilité d'avoir eu NON aux $i$ premières |
| `P = surv[..., -1]` | $P_k$ |
| `E = 1.0 + (surv[..., :-1] * valid[..., 1:]).sum(2)` | $E_k$ : la première question est toujours posée (le `1`), la suivante si les précédentes ont donné NON |
| `ratio = np.where(C, P / E, 0.0)` | $\rho_k$, mis à 0 pour les hypothèses mortes |

### Étape 4 — Choisir la cible

$$
k^* = \arg\max_{k \in \text{alive}} \rho_k
$$

C'est la règle de Smith de la section 2.3 : si les hypothèses ne partageaient
aucun bottleneck, on les traiterait dans l'ordre des $\rho_k$ décroissants, et
$k^*$ serait la première.

Code : `k = np.where(C, ratio, -np.inf).argmax(1)`.

### Étape 5 — Choisir la question

Toute question de $U_{k^*}$ fait avancer la cible. Pour départager, on regarde
à qui d'autre elle sert : un NON sur $s$ fait avancer toutes les hypothèses
vivantes qui ont $s$ dans leur $U_k$. On donne à chaque bottleneck la **masse**
des hypothèses qu'il fait avancer, pondérées par leur rentabilité :

$$
\text{masse}(s) = \sum_{k \in \text{alive},\ s \in U_k} \rho_k ,
\qquad
s^* = \arg\max_{s \in U_{k^*}} \text{masse}(s)
$$

En cas d'égalité, on prend le $s$ au $p_s$ le plus grand.

Code :
- `mass = np.einsum("bmn,bm->bn", U, ratio)` calcule $\text{masse}(s)$ pour
  tous les $s$ d'un coup (une somme des lignes de `U` pondérées par `ratio`) ;
- `score = np.where(U[rows, k], mass + 1e-9 * p, -1e8)` garde la masse sur
  $U_{k^*}$, avec le petit terme `1e-9 * p` pour les égalités, et met $-10^8$
  partout ailleurs.

La politique ne choisit pas elle-même : elle renvoie ce score, et
`evaluate_policy_on_real_human` masque les bottlenecks déjà posés à $-10^9$
puis prend l'argmax. C'est pourquoi le hors-cible vaut $-10^8$ : plus bas que
toute la cible, plus haut que le déjà-posé.

**Cas dégénéré.** Si tous les $\rho_k$ sont nuls (un $p_s = 1$ dans chaque
$U_k$), aucune cible n'a de sens et le code retombe sur la règle **qvalue** de
`policy_search/` (le bloc `fall`). Cela n'arrive jamais avec l'a priori
homogène du pipeline.

**Coût.** Un tri par hypothèse, soit $O(|\Phi| \cdot |B| \log |B|)$ par
question, sans table en $3^n$.

### Les rails sont intégrés

- Un bottleneck de **tier 1** (dans aucune hypothèse vivante) est dans tous les
  $U_k$ : il est dans $U_{k^*}$ et sa masse est maximale, donc il est demandé en
  premier.
- Un bottleneck de **tier 3** (dans toutes) n'est dans aucun $U_k$ : il n'est
  jamais candidat.

Avec ou sans `--rails`, les comptes sont identiques.

### Pourquoi la question la plus partagée ?

La règle de Smith « pure » demanderait dans $U_{k^*}$ le bottleneck le plus
risqué (variante `ratio_ff`, un peu moins bonne). Choisir le plus partagé a
deux avantages :
1. un NON fait avancer plusieurs certificats ; si la cible tombe, le travail
   n'est pas perdu ;
2. un bottleneck qui manque à beaucoup d'hypothèses est un bottleneck que peu
   d'humains ont : $p_s$ faible, donc peu de OUI inutiles.

### Dans le pipeline : a priori homogène ½

`experiment.py` appelle `solve_query_mdp_ratio(I, B)` sans `p`, donc avec
$p_s = \tfrac12$ partout (`self.p = np.full(n, 0.5)`). C'est la variante
**ratio_shared_u** des `.md`, qui n'utilise que $\Phi$ et $B$, le même budget
que H1 et H3. Les formules se simplifient :

$$
P_k = 2^{-r_k}, \qquad E_k = 2\,(1 - 2^{-r_k}), \qquad
\rho_k = \frac{1}{2\,(2^{r_k} - 1)} .
$$

$\rho_k$ décroît avec $r_k$ : la cible est simplement **l'hypothèse vivante à
qui il manque le moins de NON**. Pour donner à la place les marginales de
l'Oracle : `solve_query_mdp_ratio(I, B, p=oracle.probs_for_raw_ids(B))`.

## 6. Limites

- **Heuristique, sans preuve d'optimalité.** La règle par ratio n'est optimale
  que pour des termes **disjoints** ; nos T_k se chevauchent massivement. Le
  recalcul après chaque réponse et le choix du bit le plus partagé sont une
  adaptation propre à ce projet, pas un résultat de la littérature.
- La seule garantie sur grandes instances est empirique : être à ≤ 0,3 de L*,
  donc à ≤ 0,3 de l'optimum. Sur des hypothèses synthétiques non structurées,
  l'écart grandit (sans VI pour savoir si c'est la politique ou la borne).
- L'a priori est un produit de marginales ; les réponses réelles d'un humain
  sont corrélées. Ratio peut donc passer très légèrement sous la VI.

## 7. Ce dont elle s'inspire

Attributions reprises de POLICIES.md, avec ses réserves.

- **Règle de Smith** : W. E. Smith, « Various optimizers for single-stage
  production », *Naval Research Logistics Quarterly*, 1956 (ordonnancement par
  ratio poids/durée).
- **Ordre optimal par P/E pour un OR de ANDs disjoints** (arbres ET-OU de
  profondeur 2) : traité, sans certitude sur l'attribution originale, dans
  D. E. Smith, « Controlling backward inference », *Artificial Intelligence*,
  1989, et R. Greiner, R. Hayward, M. Jankowska, M. Molloy, « Finding optimal
  satisficing strategies for and-or trees », *Artificial Intelligence*, 2006.
  Problème voisin : H. A. Simon, J. B. Kadane, « Optimal problem-solving
  search: all-or-none solutions », *Artificial Intelligence*, 1975.
- **Cadre SBFE / sequential testing** : T. Ünlüyurt, « Sequential testing of
  complex systems: a review », *Discrete Applied Mathematics*, 2004.
- **Repli qvalue** : inspiré de A. Deshpande, L. Hellerstein, D. Kletenik,
  « Approximation algorithms for stochastic Boolean function evaluation and
  stochastic submodular set cover », SODA 2014 (adaptation, pas une réplique).

# Appendix A — Ratio as Optimistic Certification

*Markdown mirror of `AAMAS-27/A-ratio-intuition.tex`, for discussion. Notation: $\mathcal{K}_\mathcal{I}$ = bottlenecks answered yes, $\mathcal{K}_\neg$ = answered no, $T_\phi = B \setminus \phi$.*

Every episode ends with a proof (Section *Hypothesis Space and Termination*). If $\mathcal{I}_G$ is achievable, the proof is a certificate: a *no* on every bottleneck of $T_\phi$ for some $\phi \supseteq \mathcal{I}_G$. If it is not, the proof is a set of *yes* answers that no hypothesis contains. A query policy therefore has two missions in one. It must **identify** which proof is available (a hypothesis that contains $\mathcal{I}_G$, or bottlenecks that no hypothesis can accommodate), and it must **collect** that proof. While the robot queries, the two missions are entangled, because it learns which proof exists only by collecting answers. They separate after the fact. The length of the shortest proof depends on the human alone, and whatever a policy asks beyond it is the price of not knowing in advance which proof to collect.

## A.1 The clairvoyant bound: the cost of the proof

**Definition 1.** For a human with implicit subgoals $\mathcal{I}_G \subseteq B$, the *clairvoyant bound* $L^\star(\mathcal{I}_G)$ is $\min\{|T_\phi| : \phi \in \Phi,\ \mathcal{I}_G \subseteq \phi\}$ if $\mathcal{I}_G$ is achievable, and $\min\{|S| : S \subseteq \mathcal{I}_G,\ S \not\subseteq \phi\ \forall \phi \in \Phi\}$ otherwise.

**Proposition 2.** Against a human with implicit subgoals $\mathcal{I}_G$, every policy asks at least $L^\star(\mathcal{I}_G)$ queries on every episode, and a policy told $\mathcal{I}_G$ in advance asks exactly $L^\star(\mathcal{I}_G)$.

*Proof.* Success requires $T_\phi \subseteq \mathcal{K}_\neg$ for a surviving $\phi$. Since $\mathcal{K}_\neg \cap \mathcal{I}_G = \emptyset$, this forces $\mathcal{I}_G \subseteq \phi$, so at least $|T_\phi| \ge L^\star$ queries were asked. Failure requires a set $\mathcal{K}_\mathcal{I} \subseteq \mathcal{I}_G$ that lies in no hypothesis, so $|\mathcal{K}_\mathcal{I}| \ge L^\star$. Conversely, asking exactly the minimising $T_\phi$, or the minimising $S$, triggers the corresponding test. ∎

Unlike the VI baseline, $L^\star$ needs no table over knowledge states. In the achievable case it is a minimum over $\Phi$. Otherwise it is $1$ if a bottleneck of $\mathcal{I}_G$ lies in no hypothesis, and $2$ if none does, because two bottlenecks of $\mathcal{I}_G$ are then incomparable (Proposition 1, cliques). The bound is thus available at every scale. Since no policy can go below it, the optimal one included, a policy's average gap to $L^\star$ is also an upper bound on its average gap to the optimum, even where the VI baseline cannot be built. $L^\star$ is a floor, not a target: reaching it on every episode would require knowing $\mathcal{I}_G$.

The gap to the bound has a simple anatomy. Let $Q$ be the number of queries of an episode that ends with $\phi_f$ certified. Every *no* on $T_{\phi_f}$ belongs to the proof, and no other answer does, so

$$
Q - L^\star \;=\; \underbrace{\bigl(|T_{\phi_f}| - L^\star\bigr)}_{\text{longer certificate}} \;+\; \underbrace{|\mathcal{K}_\mathcal{I}|}_{\textit{yes}\text{ answers}} \;+\; \underbrace{|\mathcal{K}_\neg \setminus T_{\phi_f}|}_{\text{unused }\textit{no}\text{s}} \qquad (1)
$$

Each term is spent on identification. On an episode that ends in failure, $Q - L^\star = (|\mathcal{K}_\mathcal{I}| - L^\star) + |\mathcal{K}_\neg|$.

## A.2 Ratio bets on the cheapest proof

With $q_i = \tfrac12$, $\rho_k = 1/\bigl(2(2^{r_k}-1)\bigr)$ decreases with $r_k$, so Ratio targets the surviving hypothesis with the fewest outstanding *no*s. Before the first query, this is a largest hypothesis, the most permissive one, with ties broken by a fixed order. Under the same prior, $P(\mathcal{I}_G \subseteq \phi_k) = 2^{-|T_{\phi_k}|}$, so the hypothesis that is cheapest to certify is also the one most likely to contain $\mathcal{I}_G$.

Ratio is optimistic: it acts as if its target $\phi_{k^\star}$ contained $\mathcal{I}_G$, and it asks only bottlenecks of $U_{k^\star}$. Three properties follow from the rule.

- **(a)** A *yes* always eliminates the target, since the bottleneck asked lies in $U_{k^\star} \subseteq T_{\phi_{k^\star}}$. A wrong guess is exposed by the first answer that contradicts it.
- **(b)** A *no* never changes the target. It lowers $r_{k^\star}$ by one, lowers every other $r_k$ by at most one, and eliminates nothing.
- **(c)** If the target contains $\mathcal{I}_G$, it is certified after exactly $r_{k^\star}$ further queries, in whatever order $U_{k^\star}$ is asked.

An episode of Ratio is therefore a sequence of **bets**. Each bet commits to one hypothesis and follows a shortest proof for it. It ends either with the certificate (the bet is won) or with a *yes* (the bet is lost). After a lost bet, Ratio targets the surviving hypothesis with the fewest outstanding *no*s.

By (c), the order inside $U_{k^\star}$ does not change the cost of a winning bet, which leaves Ratio a free choice. It spends that choice on the other hypotheses: it asks the bottleneck of $U_{k^\star}$ that lies in the most outstanding certificates, each weighted by its $\rho_k$. Since $\rho_k$ at least halves with every additional outstanding *no*, the weight concentrates on the runner-up hypotheses, which are the closest to certification and the likely fallbacks. A *no* on that bottleneck advances their certificates along with the target's. A *yes* eliminates them together with the target. Either way, if the bet is lost, the next target has already absorbed the *no*s collected so far.

**Proposition 3.** Let $\mathcal{I}_G$ be achievable, and let Ratio lose $b$ bets and certify $\phi_f$. Then

$$
Q_{\text{Ratio}} - L^\star = b + |\mathcal{K}_\neg \setminus T_{\phi_f}| + \bigl(|T_{\phi_f}| - L^\star\bigr) \;\ge\; b.
$$

In particular, (i) $Q_{\text{Ratio}} = L^\star$ iff the first target contains $\mathcal{I}_G$, and (ii) $Q_{\text{Ratio}} = L^\star + 1$ iff exactly one bet is lost, every *no* collected before it lies in the certificate of the second target, and that certificate has size $L^\star$.

*Proof.* By (a), every *yes* ends a bet and every lost bet ends with a *yes*, so $|\mathcal{K}_\mathcal{I}| = b$, and the identity is Equation (1). For (i), if the first target $\phi_1$ contains $\mathcal{I}_G$, every query is answered *no*, so by (b) and (c) the episode ends after $|T_{\phi_1}| = \min_\phi |T_\phi| \le L^\star$ queries. Otherwise $T_{\phi_1}$ holds a bottleneck of $\mathcal{I}_G$, which must be asked before $\phi_1$ can be certified. No other hypothesis can be certified first, because $\phi_1$ keeps the fewest outstanding *no*s throughout the bet. The first bet is therefore lost, and $b \ge 1$. Statement (ii) is the identity with $b = 1$ and the other two terms zero. ∎

## A.3 What the episodes show

**Table 3.** Average gap to the clairvoyant bound, $Q - L^\star$, pooled over the four domains, and the distribution of Ratio's gap. $N$ is the number of episodes. The VI baseline cannot be built in the large setting.

| Setting | $N$ | $L^\star$ | Random | Info Gain | Query Freq. | VI | **Ratio** | Ratio $= L^\star$ | $= L^\star{+}1$ | $\ge L^\star{+}2$ |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact, $\lvert\mathbb{M}^H\rvert = 20$ | 157 | 2.59 | 7.45 | 4.28 | 1.50 | 0.17 | **0.17** | 84.7% | 14.0% | 1.3% |
| Large, $\lvert\mathbb{M}^H\rvert = 100$ | 200 | 14.77 | 26.73 | 23.64 | 2.96 | – | **0.46** | 84.0% | 9.5% | 6.5% |
| Large, $\lvert\mathbb{M}^H\rvert = 200$ | 200 | 22.72 | 36.09 | 32.52 | 4.52 | – | **0.56** | 72.5% | 13.5% | 14.0% |

Table 3 gives the average gap of every condition. Ratio stays within $0.17$–$0.56$ queries of the cheapest proof, that is within $2.5$–$6.4\%$ of $L^\star$. Query Frequency needs $1.5$–$4.5$ extra queries, and Info Gain and Random need $4$–$36$. In the large setting, where $L^\star$ is the only reference available, it places Ratio within $0.46$ and $0.56$ queries of the optimal policy on average.

In the exact setting, we replayed the $157$ Ratio episodes from their seeds, recovering every query count, and decomposed each one with Proposition 3. The first target contains $\mathcal{I}_G$ in $128$ of the $152$ episodes where $\mathcal{I}_G$ is achievable, and Ratio ends those in exactly $L^\star$ queries. In the other $24$, Ratio loses one bet ($22$ episodes) or two ($2$ episodes). In every one of them, the final certificate has size $L^\star$ and contains all the *no*s collected during the lost bets ($27$ of $27$). The gap is thus exactly the number of lost bets: each wrong guess cost a single *yes*, and nothing collected before it was wasted. Ties explain many wrong first guesses. In $13$ of the $24$ episodes, another hypothesis of the same size as the first target contained $\mathcal{I}_G$. Under the prior, the two are indistinguishable until a query separates them. The VI baseline, optimal in expectation under the same prior, has the same gap distribution (a gap of $0$, $1$ and $2$ on $133$, $22$ and $2$ of the $157$ episodes), although not on the same episodes. The gap is therefore not an artifact of Ratio's greedy rule: the policy that is optimal for the prior pays it too.

In the large setting, we replayed every episode with an achievable $\mathcal{I}_G$ and a gap of two or more ($9$ with $100$ candidate humans, $22$ with $200$), together with $10$ control episodes per setting, and every count again matched. The final certificate had size $L^\star$ in all of them: the fallback was always an optimal one. All but two of these $31$ episodes lost a single bet. Their extra queries are the $1$ to $4$ *no*s, collected during the lost bet, that the fallback's certificate did not contain. Over the replayed episodes, $91\%$ of the *no*s collected during lost bets were reused ($567$ of $621$).

**When $\mathcal{I}_G$ is not achievable.** The proof is then a set of *yes* answers that no hypothesis contains. In all $24$ such episodes of the three settings, the human wanted a bottleneck that lies in no hypothesis, a tier-1 bottleneck in the sense of Section *Rails*, so $L^\star = 1$. Ratio asks the tier-1 bottlenecks first, because they lie in every outstanding certificate. It stops at the first *yes*, which is already a minimal proof. Every query before it is a *no* on a tier-1 bottleneck, which any success certificate would also have required; replaying the $19$ unachievable episodes of the large setting confirmed this on every one of them. Ratio's gap on these episodes is therefore the number of tier-1 bottlenecks it asks before reaching one the human wants. Under the homogeneous prior, the tier-1 bottlenecks are interchangeable, so no rule that sees only $\Phi$ and $B$ can find that bottleneck faster in expectation. The VI baseline breaks the tie the same way: on $493$ small instances solved exactly, its optimal actions were exactly the unasked tier-1 bottlenecks in every knowledge state that had one. Unachievable episodes are rare ($5\%$ of the large setting), but on boards with many tier-1 bottlenecks they are the costliest. With $100$ candidate humans, two Taxi episodes needed $15$ and $26$ queries against $L^\star = 1$. The unachievable episodes account for $47\%$ of Ratio's total gap with $100$ candidate humans and $25\%$ with $200$.

Ratio thus never sets out to identify $\mathcal{I}_G$: it identifies only as a by-product of trying to prove. Its guesses are cheap because the hypothesis it bets on is both the cheapest to certify and the most likely to be right. A wrong guess costs one *yes*. The free order inside each certificate prepares the next guess.


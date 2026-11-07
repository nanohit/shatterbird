# Phase 0 — Theory, closures and pre-registered predictions

*Written before any experiment was run (spec §2: "targets are written before runs"). Later revisions are marked **[rev]** with a reason.*

This note does three things the spec requires before implementation:

1. derives what E1 must show, so the implementation can be validated against closed forms instead of eyeballed;
2. closes the model definitions the spec leaves open (§3.2 interface law, §4.1 continuation/event rules, §4.3 S2 contract, time integrators, energy ledger);
3. states predictions for E2–E10 that the runs can falsify.

Assignment interpretation. The spec's §0 handoff says "E1 only"; the author has since asked for all Phase 0 stages. The order follows §5 and the §0 gating: E1 → E4/E2 → E3 → (law closure, §3 below) → E5 → E8 → E6 → E7 → E9 → E10. Track B (§8, WebGPU device microbenchmarks) is a separate, independent track and is not part of this mechanics program. Everything here is CPU f64; no fixed point, GPU, determinism or performance claims.

---

## 1. E1 in closed form

### 1.1 Setup

One DOF $x$, two springs with constraint functions $C_1=C_2=x$ (rest length zero), physical stiffnesses $k_i$, penalties $\rho_i$, applied force $P$. Quasi-static: $E_0(x)=-Px$.

$$s_i=\frac{k_i}{k_i+\rho_i},\qquad w_i:=k_{\mathrm{eff},i}=\rho_i s_i,\qquad W=w_1+w_2,\qquad \sigma=s_1+s_2.$$

Schedule (spec §4.2, last bullet): exact primal minimisation with $\lambda$ frozen, then one simultaneous dual update.

$$x^+=\frac{P-\textstyle\sum_i s_i\lambda_i}{W},\qquad \lambda_i^+=s_i(\lambda_i+\rho_i x^+).$$

Fixed point: $\lambda_i^*=k_ix^*$, $x^*=P/(k_1+k_2)$. Note $\lambda_i^+=s_i\lambda_i+w_ix^+$ is exactly the force spring $i$ exerts in the primal objective at $x^+$, so after every update $\sum\lambda^+=P$ (primal equilibrium) — **force balance is not evidence of convergence**.

### 1.2 Iteration matrix (general $\rho_i$)

With $e=\lambda-\lambda^*$: $e^+=Me$, $M=S-\dfrac{w\,s^\top}{W}$, $S=\mathrm{diag}(s)$.

- $\mathbf 1^\top M=s^\top-s^\top=0$: left eigenvector $\mathbf 1$, eigenvalue 0 → the sum error is annihilated in one update for **any** penalty policy.
- $M\rho=w-w=0$: the right eigenvector of eigenvalue 0 is the vector of $\rho_i$ (equal to $\mathbf 1$ only for shared $\rho$).
- The other eigenvalue is $\operatorname{tr}M$:
$$r=\frac{s_1w_2+s_2w_1}{w_1+w_2},\qquad\text{right eigenvector }u=(1,-1)/\sqrt2\ (\perp\mathbf 1).$$
- Shared $\rho$: $r=\dfrac{2s_1s_2}{s_1+s_2}=\dfrac{2}{2+\rho/k_1+\rho/k_2}$ (context §6.4). Proportional $\rho_i=c\,k_i$: $s_1=s_2=1/(1+c)$ and $r=1/(1+c)$.

So after the first update the whole error is $d_n u$ with $d_{n+1}=r\,d_n$ exactly. This gives exact predictions for every measured quantity (validated in E1 to round-off):

| quantity | after iteration $n\ge1$ (error $e_n=d_nu$) |
|---|---|
| $\sum\lambda-P$ | $0$ |
| $x_{n+1}-x^*$ | $-s^\top e_n/W=-d_n(s_1-s_2)/(\sqrt2\,W)$ |
| constitutive residual $\lambda_i-k_ix$ | $e_{n,i}-k_i(x-x^*)$ |
| complementary energy $\sum\lambda_i^2/2k_i-U^*$ | $\sum_i e_{n,i}^2/2k_i\ \ge0$ (exactly quadratic, because $\sum e=0$) |
| displacement energy $\tfrac12(k_1+k_2)x^2-U^*$ | first order in $x-x^*$ |

Remarks that matter for the interpretation:
- ~~Position error is small exactly when $s_1\approx s_2$ … positions look converged while forces are not.~~ **[rev, after E1]** Wrong as a relative statement. The *absolute* position error carries the factor $s_1-s_2\to0$, but $x^*=P/(k_1+k_2)$ is equally small. For $k\gg\rho$ the relative position error is $\approx\frac{k_2^2-k_1^2}{2k_1k_2}\,|c|/P$, **independent of $\rho$** and of the same order as the distribution error: 12.5 % for $k_2/k_1=2$, ≈200 % for $k_2/k_1=10$ at iteration 8. Both decay at $r$. Positions are not a safe proxy either.
- $r$ depends only on the ratios $\rho_i/k_i$. Fast distribution needs $\rho_i\gtrsim k_i$ for both springs. In E1 that is free, because the primal is a scalar solved exactly; in any multi-DOF problem it moves the physical stiffness into the primal and its conditioning. **E1 cannot measure that cost** (limit of the counterexample; it is measured in E2/E3).

### 1.3 Regimes, predicted

Let $\kappa=k/(k_1+k_2)$ (so $\lambda^*=\kappa P$) and $\delta=u^\top\kappa\,\Delta P$.

- **Cold start** ($\lambda=0$): $e_0=-\kappa P$; after one update $d_1=-r\,u^\top\kappa P$. With $s_1\approx s_2$ the split is $\approx P/2:P/2$.
- **Linear ramp with warm start**, $n$ steps of $\Delta P$, $m\ge1$ iterations per step, starting converged at $P=0$:
$$d_n=-\delta\,\frac{r^m(1-r^{mn})}{1-r^m},\qquad \frac{|d_n|}{|u^\top\lambda^*_n|}=\frac{r^m(1-r^{mn})}{n(1-r^m)}\ \approx\ \frac{1}{n\,m\,(1-r)}\quad(nm(1-r)\gg1).$$
  A warm-started ramp is only as good as its *total* iteration count $nm$ measured against $1/(1-r)$. Prediction for $k_1{=}10^3\rho$, ratio 1:2, 60 Hz, $m=8$: a 10 s ramp still has ≈28 % distribution error.
- **Step change in $P$** (converged at $P_a$, new $P_b$): $d_j=r^j u^\top\kappa(P_a-P_b)$.
- **Step change in $k_2$** (damage stand-in; new $r'$). Three declared multiplier policies, all predicted:
  - *keep* $\lambda$ (force continuity): $e_0=P(\kappa_{\rm old}-\kappa_{\rm new})$, already $\perp\mathbf 1$, $d_j=r'^jd_0$;
  - *rescale* $\lambda_2\leftarrow\lambda_2k_2'/k_2$ (strain continuity): $e_0=k'(x^*_{\rm old}-x^*_{\rm new})$, first update removes its $\mathbf 1$-component, then $r'$;
  - *reset* $\lambda=0$: cold start with $r'$.

### 1.4 S2 in E1

Quasi-static, linear, scalar: $Ku=P$ with $K=k_1+k_2$ is S0. S2 is exact after one $1\times1$ factorization. Two declared couplings: (A) auxiliary field — forces $k_iu$ reported, motion $x$ untouched; (B) dual injection — $\lambda_i\leftarrow k_iu$, after which S1 is at its fixed point. E1 therefore carries **no information about S2's cost or coupling consistency**; it only checks the plumbing.

### 1.5 Dynamic extension (labelled E1-dyn, not in the spec)

Adding $\tfrac{\mu}{2}(x-\tilde x)^2$, $\mu=m/\Delta t^2$, gives $M_\mu=S-ws^\top/(W+\mu)$: the sum error is no longer killed in one update, and the slow mode stays $\approx1-O(\rho/k)$. Inertia does not rescue force distribution. One quick run documents the recommended $\rho\approx m_{\rm eff}/\Delta t^2$ policy in this setting.

---

## 2. Why local solvers face an elliptic lower bound (prediction for E2/E3/E7)

For an implicit step, the system matrix is $M/\Delta t^2+K$. Its **screening length** in cells is

$$\frac{L_s}{\ell}\approx\sqrt{\frac{k\,\Delta t^2}{m}}\approx\frac{c\,\Delta t}{\ell},\qquad c=\sqrt{E/\rho_{\rm d}}\approx3.5\ \text{km/s (reference concrete)}.$$

| cells | $\Delta t$ | $c\Delta t/\ell$ |
|---|---|---|
| 0.5 m | 1/480 s | ≈15 |
| 0.5 m | 1/1920 s | ≈3.7 |
| 1 cm | 1/480 s | ≈740 |

Within $L_s$ the step is a quasi-static (elliptic) problem. Any method that exchanges information only between neighbouring cells needs at least $\sim L_s/(\ell\cdot p)$ sweeps to propagate a change across it ($p$ = cells per sweep a multicolour Gauss–Seidel can carry, a small number). Unaccelerated Gauss–Seidel/ALM needs $\sim(L_s/\ell)^2$. Compliant ALM rescales the primal blocks but cannot shorten the domain of dependence. In quasi-statics there is no inertia at all, so with uniform stiffness a small shared $\rho$ does not even improve primal conditioning ($\rho\sum J^\top J$ has the same condition number as $K$); ALM's gain is equalising stiffness *ratios*.

Predictions (to be falsified):
- **P-E2.** On a clamped block ≈10–15 cells across, quasi-static cold start: no S1 policy and not S3 reaches 1 % facet-force error within 16 sweeps. Error after a fixed budget gets worse as $\rho/k$ decreases (dual-limited) and also does not reach 1 % for $\rho\ge k$ (primal-limited).
- **P-E3a.** Sweeps to 1 % grow ≈ quadratically with column height $N$ for S1/S3. With block-Jacobi-preconditioned CG they grow ≈ linearly (a reference curve for "global iterative").
- **P-E3-dyn.** In dynamics, sweeps to 1 % scale with $(c\Delta t/\ell)^2$ until $L_s$ exceeds the structure size. Smaller substeps therefore cost less *total* sweep work per frame for Gauss–Seidel-type solvers: work/frame ∝ $(1/\Delta t)\cdot\Delta t^2$.
- **P-S2.** Quasi-static linear S2 equals S0 (same equations) and is exact at the cost of one factorization. In dynamics, S2's force error is $M\,\delta a$ instead of $K\,\delta q$, about $m/(k\Delta t^2)$ times smaller for the same motion error.

---

## 3. Model closure (2D)

### 3.1 Geometry and DOFs

Seeds by random sequential addition (min distance $d_{\min}$, deterministic xoshiro256** RNG). Voronoi cells come from half-plane clipping of the convex domain, with the stop rule "next candidate seed farther than twice the current maximum vertex radius". Cells are rigid laminae of thickness $b$; DOFs $(x,y,\theta)$ at the **centroid** $X$ (mass $m=\rho_{\rm d}Ab$, polar inertia about the centroid). Generators $g$ are stored and define $n$ and $\ell_e$. The arms $a=p-X$ are taken from the centroid to the material point; generator-referenced uniformity is obtained by evaluating the rigid motion at the facet points, never by substituting distances.

Supports are **ground facets**: a boundary edge of a cell connects to a prescribed rigid body, with the normal pointing from the cell into the support. **[rev, before E4]** $\ell_e=\mathrm{dist}(g,\text{edge line})$. The mirror distance $2d$ would be correct only for a symmetry plane, whose mirrored body deforms symmetrically. Against a rigid support it halves the boundary stiffness, and uniform strain would no longer be an exact solution. Removing a support = removing a ground facet. Notches = facets whose segment intersects the notch segment are removed.

### 3.2 Kinematics: two-point corotated facet

Each facet carries two integration points at $p_\pm=p_c\pm\frac{l}{2\sqrt3}t$, each with half the facet area. At point $p$ (reference), with $R(\theta)$ the exact rotation:

$$\Delta=\big(x_b+R(\theta_b)(p-X_b)\big)-\big(x_a+R(\theta_a)(p-X_a)\big),\qquad \delta=R(-\bar\theta)\Delta-\delta_{\rm rest},\qquad \bar\theta=\tfrac12(\theta_a+\theta_b),$$
$$\delta_n=n_0\!\cdot\!\delta,\qquad \delta_t=t_0\!\cdot\!\delta .$$

Properties:
- **[rev, E6]** For a facet against a support or plate body the frame is the support's own, $\bar\theta=\theta_b$ (general form $\bar\theta=(1-w)\theta_a+w\theta_b$, $w=\tfrac12$ between cells, $w=1$ against supports). In the mean frame a roller's normal tilts by $\theta_{\rm cell}/2$ and the roller then resists sliding: in E6 this gave a spurious arch.
- Rigid motion of a pair gives $\Delta\equiv0$ exactly (objectivity), and a loaded pair rotated rigidly gives the same $\delta$.
- With per-point stiffness $k_n/2,\ k_t/2$, linearisation gives exactly $\tfrac12k_n\delta_{n,c}^2+\tfrac12k_t\delta_{t,c}^2+\tfrac12\,(k_nl^2/12)\,\delta_\phi^2$, because $(l/2\sqrt3)^2=l^2/12$. **The two-point facet is the spec's three-spring elastic baseline** ($k_n=Ebl/\ell_e$, $k_t=\gamma k_n$, $k_\phi=k_nl^2/12$). It realises bending as differential normal opening, which is what makes the fracture law below energy-consistent under rotation.
- $\delta_{\rm rest}=0$ for exact geometry; the field exists so that quantised geometry can be represented the same way later.

### 3.3 Interface law SB-IL1 (per integration point) — *candidate, not a literature formulation*

Parameters: $E,\sigma_t,G_c$, $\gamma=k_t/k_n$, facet $\ell_e$, point area $A_p=lb/2$, point stiffnesses $k_n^p=EA_p/\ell_e$, $k_t^p=\gamma k_n^p$.

- Equivalent opening (energy norm): $\delta_{\rm eq}=\sqrt{\langle\delta_n\rangle_+^2+\gamma\,\delta_t^2}$.
- Onset and final openings: $\delta_0=\sigma_t\ell_e/E$, $w_c=2G_c/\sigma_t$. Admissibility requires $\delta_0<w_c$, i.e. $\ell_{\rm cb}=\ell_e<2l_{ch}$; checked per facet, and with the option off a violation is an error.
- History: committed $\kappa_c$; trial $\kappa=\max(\kappa_c,\delta_{\rm eq})$; damage
$$g(\kappa)=\begin{cases}0&\kappa\le\delta_0\\ \dfrac{\kappa-\delta_0}{\kappa(1-\delta_0/w_c)}&\delta_0<\kappa<w_c\\ 1&\kappa\ge w_c\end{cases}$$
- Stored energy: $\psi=\tfrac12(1-d)\big(k_n^p\langle\delta_n\rangle_+^2+k_t^p\delta_t^2\big)+\tfrac12k_n^p\langle\delta_n\rangle_-^2$ — undamaged compressive branch, shear degrades.
- Forces: $f_n=(1-d)k_n^p\langle\delta_n\rangle_++k_n^p\langle\delta_n\rangle_-$, $f_t=(1-d)k_t^p\delta_t$. Secant unloading to the origin; crack opening $w=d\,\delta_n$ in tension, so a crack closes at $\delta_n=0$ and transmits pressure beyond it.
- Dissipation: $\dot D=Y\dot d$, with $Y=\tfrac12k_n^p\delta_{\rm eq}^2=\tfrac12k_n^p\kappa^2$ while damage grows, which integrates in closed form to
$$D(\kappa)=\tfrac12k_n^p\,\delta_0\,\frac{\kappa-\delta_0}{1-\delta_0/w_c},\qquad D(w_c)=\tfrac12k_n^p\delta_0w_c=G_cA_p .$$
  Because $d$ depends only on $\kappa$ and $Y=\tfrac12k_n^p\kappa^2$ on the loading branch, **total dissipation to full damage is $G_cA_p$ for every path and mode mix** (not only pure normal). Partial damage dissipates $D(\kappa)$ — linear in $\kappa$, not in $d$.
- Loading-branch potential: for $\delta_{\rm eq}\ge\kappa_c$, $\boldsymbol f=\partial\Psi/\partial\boldsymbol\delta$ with $\Psi=\int_0^{\delta_{\rm eq}}(1-g(s))k_n^ps\,ds$, so **the consistent tangent is symmetric** ($\alpha=\gamma$ is what makes it so):
$$\mathbf D_t=(1-d)\mathbf K-\frac{g'(\kappa)}{k_n^p\,\delta_{\rm eq}}(\mathbf K\hat{\boldsymbol\delta})(\mathbf K\hat{\boldsymbol\delta})^\top,\quad \hat{\boldsymbol\delta}=(\langle\delta_n\rangle_+,\delta_t),\ \mathbf K=\mathrm{diag}(k_n^p,k_t^p).$$
  The trial problem within a step is therefore an incremental *energy minimisation* (non-convex once softening), which S0 and S1 both address.
- Known limitations (named, not hidden): shear strength and mode-II energy are tied to mode I through $\gamma$; no pressure-dependent shear strength or friction; damage drives shear in compression; geometric facet area (not projected crack area) normalises $G_c$. On a random lattice a macroscopic crack crosses inclined facets, so expected macroscopic dissipation is $G_c\times$ ligament area $\times$ a tortuosity factor $>1$. E6 measures it, and Bolander–Berton's objectivity results are **not** inherited.
- Crack-band option (off by default, always reported): $\sigma_{\rm eff}=\min\!\big(\sigma_t,\ \eta\sqrt{2EG_c/\ell_e}\big)$ with $\eta<1$ (default 0.95). This keeps a finite softening slope; $\eta=1$ is the degenerate elastic-brittle drop and is refused. It changes strength, so it is a coarse-graining model, not a stability guarantee.

### 3.4 Trial / commit

Inside a load or time step every evaluation uses $\kappa=\max(\kappa_c,\delta_{\rm eq}(q_{\rm trial}))$. Solver iterations never write $\kappa_c$. On acceptance of the step: $\kappa_c\leftarrow\kappa$ and the dissipation ledger adds $D(\kappa)-D(\kappa_c)$. On rejection (non-convergence, event-size violation) nothing is written. A one-shot damage update after a frozen-damage solve is a separately labelled *lagged* variant.

---

## 4. Solvers

### 4.1 S0

Newton on the (incremental) potential with the **Gauss–Newton material tangent** $\sum J^\top\mathbf D_tJ$. The geometric second-derivative terms of the corotated kinematics are omitted. They are $O(\text{force}/\text{stiffness})=O(\text{strain})$ relative, so convergence stays effectively quadratic at strains $\sim10^{-4}$, and the reference is defined by its converged residual, not by its Hessian. Sparse Cholesky (faer, AMD) for SPD systems, sparse LU for indefinite or bordered ones. Tolerance: $\|r\|_\infty\le10^{-10}\times$(load scale) plus step norm. Backtracking line search on the energy for the non-convex trial problem.

Continuation for softening (E5/E6):
- **Control:** prescribed displacement (E5) or CMOD via a bordered system $[\,H\ \ -\hat f;\ c^\top\ \ 0\,]$ (E6).
- **Acceptance:** Newton converged and $\max\Delta d\le\Delta d_{\max}$ over all points; otherwise the step is halved (no history written).
- **Events / ties:** S0 solves the **coupled** damage law (all trial points simultaneously, backward-Euler in $\kappa$), *not* Bolander–Berton's sequential one-spring-set-per-cycle algorithm. Simultaneity is controlled by $\Delta d_{\max}$ and checked by step refinement ($\Delta d_{\max}\in\{0.2,0.1,0.05\}$). A converged Newton state is *an* equilibrium; branch admissibility is assessed by refinement agreement and the energy ledger, not by the residual.

### 4.2 S1 (as specified)

Per-cell $3\times3$ block; one Gauss–Newton step per block visit on $\Phi(q;\lambda)$ with Hessian $\mu_iM_i+\sum_ck_{\mathrm{eff},c}J_c^\top J_c$. Deterministic greedy colouring in cell-index order. Colour-sequential sweep: all cells of a colour read the state left by the previous colours. On a CPU, sequential updates inside a colour are identical to a parallel snapshot update, because a colour has no internal facets. One primal sweep, then one simultaneous dual update on the completed state. Scalar components $c$ = (point, normal|tangent). Penalty policies: shared $\rho=c\,\bar k_n$; per-component $\rho_c=c\,k_c$; dynamic $\rho_c=m_{{\rm eff},c}/\Delta t^2$; **S3** = the $\rho\to\infty$ limit ($s=0$, $k_{\rm eff}=k$: plain penalty block descent, no dual). Damage stiffness enters as the trial secant $k_c=(1-d)k_c^0$ (tension) per sweep, from committed history.

Forces reported for S1 are **both** $\lambda$ (dual) and $k_cC_c(q)$ (constitutive), plus their difference.

### 4.3 S2 contract

- **Equations.** Linear, reference-frame stiffness $K$ assembled from the current trial secant law (frozen during the step). Quasi-static: $Ku=f_{\rm ext}$ (identical to S0 for a linear problem). Dynamic: $Ku=f_{\rm ext}-Ma(q_{S1})$, with $a$ the integrator's acceleration of the S1 motion.
- **Role of $u$.** Auxiliary elastic field. It defines the reported forces $F=K_eu$. Motion, external work and measured displacements come from $q_{S1}$. **Feedback (B, "dual injection")**: optionally $\lambda_c\leftarrow F_c(u)$ before further S1 sweeps. The exact solution is a fixed point of (B), so (B) is a consistent hybrid whose convergence is measured, not assumed.
- **Free clusters.** Components are found from the active facet graph. For a component with no prescribed body, the load $b=f-Ma$ is made self-equilibrated by projecting out its rigid part (mass-weighted). $Ku=b$ is solved with the rigid modes fixed by a bordered gauge constraint $R^\top Mu=0$: a gauge, not a support, and its multipliers vanish for compatible loads. Mechanisms after separation appear as extra zero pivots. They are detected and reported, never regularised away.
- **Factor reuse.** Symbolic analysis is reused while the facet graph is unchanged. Numeric factors are reused only while no secant stiffness changed and the reference-frame approximation holds (cluster mean rotation $<$ 1°). Otherwise refactor. Symbolic analyses, numeric factorizations, triangular solves, nnz(L) and flop estimates are counted separately.
- S2 is labelled *reduced/partitioned* whenever $q_{S1}$ is not converged; its extra error is measured against S0 in displacement, acceleration, resultants, work, energy and damage.

### 4.4 Time integrators (all variational: minimise $\tfrac{\alpha}{2\Delta t^2}\|q-\hat q\|_M^2+\Pi(q)$)

| scheme | $\alpha$ | $\hat q$ | per-step amplification on $\ddot x=-\omega^2x$, $\Omega=\omega\Delta t$ |
|---|---|---|---|
| implicit Euler | 1 | $q_n+\Delta t\,v_n$ | $\lvert z\rvert=(1+\Omega^2)^{-1/2}$ |
| BDF2 | 9/4 | $\tfrac43q_n-\tfrac13q_{n-1}+\tfrac{2\Delta t}{3}(\tfrac43v_n-\tfrac13v_{n-1})$ | root of $(3-2i\Omega)z^2-4z+1=0$ |
| Newmark avg. accel. | 4 | $q_n+\Delta t\,v_n+\tfrac{\Delta t^2}{4}a_n$ | $\lvert z\rvert=1$, phase $2\arctan(\Omega/2)$ |

With $\Omega=2\pi\cdot50\,\text{Hz}/480$: implicit Euler keeps $(1+\Omega^2)^{-9.6/2}\approx0.17$ per period (context §2.1). Prediction **P-E8**: implicit Euler at 1/480 s removes most of the response of any mode above ~10 Hz within a few periods. A 10 Hz mode loses ~⅓ of its amplitude per period. BDF2 is much better below ~50 Hz; Newmark has no amplitude loss and only period error.

### 4.5 Energy ledger

$\Delta(T+U_{\rm el}+V_g)+D_{\rm frac}+D_{\rm num}=W_{\rm ext}+\varepsilon$, with
- $W_{\rm ext}$ = trapezoidal work of applied and reaction forces on prescribed motion;
- $D_{\rm frac}$ = the law's closed-form $D(\kappa)$ increments (§3.3), independent of the integrator;
- $D_{\rm num}$ (implicit Euler) = $\tfrac12\|v_{n+1}-v_n\|_M^2+\big[\nabla\Pi(q_{n+1})\cdot\Delta q-\Delta\Pi\big]_{d=d_{n+1}}$ (kinetic and Bregman terms of the scheme);
- $\varepsilon$ = **signed** remainder, never relabelled as numerical dissipation.

---

## 5. Pre-registered expectations for the rest

- **E4.** With $\gamma=1$ and boundary cells driven by the uniform field evaluated at the facet points, interior tractions equal $E\varepsilon n$ on every facet to round-off, for any tessellation (exact uniformity). Driving at centroids instead of the rigid motion of generators breaks it at $O(\|g-X\|/\ell)$. Uniaxial tension: $E_{\rm eff}=E$, $\nu_{\rm eff}=0$. For $\gamma\ne1$ the affine (Voigt) estimate is $\nu=(1-\gamma)/(3+\gamma)$, $E_{\rm eff}=E(3+\gamma)(1-\nu^2)/4$, and relaxation makes the actual values lower. Rigid rotations up to 180° and translations: spurious forces at round-off, growing with $|x|\cdot\epsilon_{64}$ for large translations.
- **E5.** Dissipated energy $\to G_clb$ (each point $G_cA_p$). The trapezoidal ledger is exact on piecewise-linear branches, so the residual is nonzero only on steps that straddle a kink; it shrinks with step refinement. Zero dissipation on unloading; $D(\kappa)$ linear in $\kappa$.
- **E6.** Peak load and dissipation agree within the 10–15 % target across 3 resolutions if the crack band is small ($\ell_e\ll l_{ch}$). Dissipation $\approx(1.1\text{–}1.3)\,G_c\,A_{\rm lig}$ from tortuosity with geometric-area normalisation.
- **E7.** S1 under 8–16 sweeps shows delayed load redistribution after support loss (many steps), S0 redistributes within the integrator's response, and S2's quality depends on the S1 acceleration.
- **E9.** Damage from impact depends strongly on $\Delta t$ when contact duration ≪ $\Delta t$. Event-triggered local time refinement recovers the fine reference.
- **E10.** The normalised 6×6 block eigenvalues respect $[1,1+\sum k_{\rm eff}\|v\|^2]$, and the bound is loose by the valence tail. Force-convergence conclusions transfer qualitatively.

---

## 6. Post-run status of the predictions **[rev, after all runs]**

| prediction | outcome |
|---|---|
| E1 closed forms | confirmed to ≤ 2·10⁻⁸; position-error remark corrected (§1.2) |
| P-E2 (no local method reaches 1 % in 16 sweeps) | confirmed: best 30 % at 16 sweeps; 75–100 sweeps needed |
| P-E3a (≈ N² growth for local methods) | **wrong exponent**: ≈ N⁴ for slender walls (bending is fourth order); PCG ≈ N |
| P-E3-dyn (work ∝ π_h; smaller steps cheaper per frame for GS) | confirmed for S3; two similar walls collapse exactly on π_h; best S1 ≈ 1 000 sweeps/frame at every h |
| P-S2 (quasi-static S2 = S0; dynamic S2 error ∝ M δa) | quasi-static confirmed; dynamic S2 fails in practice because δa from a budgeted S1 is large (and often unstable), see E7 |
| E4 uniformity / objectivity | confirmed (2·10⁻¹¹; no drift); ν above Voigt (prediction said below) |
| E5 energy closure | confirmed (exact G_cA; ledger → 0 with refinement) |
| P-E8 (IE kills ≳10 Hz modes at 1/480) | confirmed; IE also halves peak forces of sudden loads → BDF2 baseline |
| E6 objectivity within 10–15 %, D ≈ 1.1–1.3 G_c A_lig | confirmed (see E6) |
| E7 "S1 delays load redistribution" | **worse than predicted**: budgeted S1 is dynamically unstable (diverges or creates energy), and S2(B) does not stabilise it |
| E9 strong h-dependence; event refinement recovers the reference | h-dependence confirmed (coarse steps: ~0 fracture); event refinement recovers 47–86 %, not all |
| E10 transfer | confirmed |

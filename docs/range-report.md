# Q5 — Range report (input for fixed-point / Flock IR design)

**Status:** *observed* magnitudes on the Phase 0 scenes, not proofs of bounds. Every number is conditional on the scene contract stated with it. Sources: `results/Q5/ranges.json` (`cargo run --release --bin ranges`), plus E1, E4, E7, E10 summaries.

**Scene contracts.**
- *Lab:* 0.2 × 0.1 m block, 5 mm cells, static patch load (0.5, −2) MPa, elastic regime.
- *Game:* E7 scenario A, an 8 × 1.2 m beam with 0.3 m cells; the 0.1 s after a support loss includes root cracking and a hinge; BDF2, $h=1/480$ s.
- Both use reference concrete ($E$ = 30 GPa, $\sigma_t$ = 3 MPa, $G_c$ = 100 N/m).

## Observed ranges

| quantity | lab (static) | game (dynamic, with fracture) | dynamic range needed | comment |
|---|---|---|---|---|
| absolute position | 3.5 mm – 0.20 m | 0.11 – 7.9 m | — | f64 absolute coordinates lose ~ε·L: at $10^6$ m the spurious forces reach $10^{-4}$ of failure (E4). **Base+delta is required.** |
| per-step translation Δx | 0.3 nm – 6 µm (one solve) | 14 nm – **2.3 mm** | ~17 bits above the 3.7 nm quantum | well inside Q4.28's ±8 m; per-step deltas are mm, not m |
| rotation (total) | ≤ 1.2·10⁻⁴ rad | ≤ 0.031 rad in 0.1 s (root hinge) | — | Q2.30 (≈10⁻⁹ rad) adequate |
| per-step rotation | — | 2·10⁻⁸ – 9.3·10⁻⁴ rad | ~15 bits | |
| facet-point normal jump δn | 10⁻¹¹ – 6.4·10⁻⁷ m | 7·10⁻¹² m – **2.9 cm** | 16 bits elastic; 32 bits with cracks | intact and cracked regimes differ by 4–5 decades; consider separate formats or a log/float-like representation for openings |
| onset stretch δ₀ = σ_t ℓ/E | **0.5 – 1.0 µm** | 30 – 59 µm | — | lab-scale δ₀ is only ~135 quanta at 3.7 nm (≈0.7 % resolution of onset); game scale ~10⁴ quanta |
| final opening $w_c$ | 66.7 µm | 66.7 µm | — | constant |
| facet strain δn/ℓe | ≤ 7.4·10⁻⁵ | ≤ 8 % (open cracks) | | |
| point stiffness $k_n^p$ | 2.7·10⁵ – 1.2·10⁹ N/m | 2.9·10⁶ – 6.2·10⁹ N/m | **11–12 bits within one mesh** | short (sliver) facets; RSA's minimum distance does not bound facet length |
| point force | 6·10⁻³ – 369 N | 0.01 N – 290 kN | 16 bits / 24 bits | |
| velocity | — | 10⁻⁵ – 1.1 m/s (after 0.1 s) | | free fall dominates later |
| ALM $s=k/(k+\rho)$ | 3.7·10⁻⁵ – 0.91 (all S1 runs logged) | | 15 bits | with $\rho\gtrsim k$ policies $s$ is small; with $\rho\ll k$, $1-s$ is small (E1: down to 10⁻⁴) |
| ALM $1-s$ | 0.09 – 1.0 (per-spring 0.1) · E1: 10⁻⁴ – 0.99 | | 13 bits (E1) | context §4: if $1-s$ rounds to 0, compliance is deleted silently. With $\rho=10^{-3}k$ that needs ≥13 fractional bits at minimum |
| $k_{\rm eff}$ | 8.2·10³ – 1.6·10¹⁰ N/m | | 21 bits | |
| multiplier λ | 1.6·10⁻¹⁴ – 2.0·10⁵ N | | ~30 useful bits (see below) | the 63-bit span comes from unloaded springs; resolution should be set relative to the failure force |
| normalised 6×6 block (3D, E10) | eigenvalues 1.9–13.2, condition ≤ 7 (ρ = m_eff/h²) | penalty: up to 4·10⁶, condition ≤ 10³ | | local conditioning is benign with bounded $k_{\rm eff}$ |
| valence (3D) | mean 15, max 19 | | | sizes fixed-size per-cell facet lists |

## Derived requirements and warnings

1. **Iteration gain on quantisation (E1).** A slowly contracting fixed-point iteration ($r\to1$) turns a per-iteration perturbation $q$ (round-off, the λ quantum) into a steady error of $\approx q/(1-r)$. At the E1 regime $r=0.99925$ that costs ~10 bits. Multiplier and position formats must be sized together with the solver's contraction, not independently.
2. **Absolute vs relative coordinates (E4).** Objectivity is exact in arithmetic, but the corotated jump subtracts absolute positions. The jump must be formed from relative quantities (base+delta), or deformation round-off grows with world size.
3. **Two regimes of jumps.** Elastic jumps (sub-µm at lab scale, tens of µm at game scale) and crack openings (mm–cm) cannot share one fixed-point format without ~32 bits plus headroom. Per-point state (intact / softening / open) could select the representation.
4. **Stiffness spread from geometry.** Sliver facets make point stiffness vary by 4 000× within a mesh. Either bound facet length in the tessellation (merge or snap short facets: a model change to validate) or keep 12+ bits of stiffness range.
5. **Inexact solves create energy (E7)** in ranges far beyond physical ones (10²⁴ J). Fixed-point range proofs are meaningless for an unstable solver: stability of the budgeted solver is a precondition for any range contract.

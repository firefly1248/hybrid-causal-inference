# Causal Effect of Hybrid Powertrain on Fuel Consumption

Estimating, from observational EPA data (~50k vehicles, model years 1984-2027), how
much fuel a hybrid powertrain actually saves once you account for the fact that
hybrids are not random - they cluster in specific segments, engine sizes, brands, and
model years. The project applies and cross-checks nine causal-inference estimators
plus a full robustness suite.

## TL;DR

- A naive comparison says hybrids get **+9 MPG (+42%)**. That is the *MPG illusion*:
  MPG is the reciprocal of consumption, so gaps at the high end are exaggerated.
- Measured correctly (gallons per 100 miles) and adjusted for confounders, comparable
  hybrids use **~0.85-0.90 fewer gal/100mi, about a 17-18% reduction in fuel** -
  roughly **130 gal and 1.1 t CO2 saved per year** at 15k mi/yr.
- Every doubly-robust estimator (AIPW, DML, DoubleML IRM, overlap weights) lands in
  the same range; within-model twins give -0.94. The result survives placebo,
  negative-control, and E-value sensitivity checks (an unmeasured confounder would
  need RR ~3.2 with both treatment and outcome to explain it away).

![Estimator comparison](05_estimator_comparison.png)

## Results (overlap population, effect on fuel use)

| Estimator | ATE (gal/100mi) | % fuel | 95% CI |
|---|---|---|---|
| Naive (unadjusted, full sample) | -1.29 | -26% | - |
| OLS adjustment | -0.73 | -15% | [-0.81, -0.64] |
| Propensity score matching (ATT) | -0.78 | -16% | [-1.00, -0.60] |
| Inverse probability weighting | -0.87 | -17% | [-1.18, -0.54] |
| AIPW (cross-fitted, doubly robust) | -0.90 | -18% | [-1.00, -0.79] |
| Double ML (CatBoost, cross-fit) | -0.85 | -17% | [-0.93, -0.78] |
| Within-model twins (Camry vs Camry Hybrid) | -0.94 | -22% | [-1.05, -0.83] |
| Overlap weights (ATO) | -0.89 | -18% | [-1.09, -0.69] |
| DoubleML IRM (library) | -0.88 | -18% | [-0.96, -0.80] |

The doubly-robust estimators cluster tightly at **-0.85 to -0.90 gal/100mi**. An EconML
causal forest agrees on sign but is imprecise here (wide CI under low treatment
prevalence), so it is reported for its point estimate and heterogeneity only.

## Method

Treatment is a hybrid (non-plug-in HEV) powertrain vs pure gasoline. Confounders:
engine displacement, cylinders, model year, forced induction (turbo/supercharger),
size class, drive type, manufacturer. Post-treatment variables (CO2, fuel cost, power)
are deliberately excluded.

Estimators: naive difference, OLS regression adjustment, propensity-score matching,
IPW (Hajek), AIPW, Double ML, within-model exact matching, overlap weights (ATO),
DoubleML IRM, EconML causal forest for heterogeneous effects.

What makes the estimates trustworthy:

- **Right outcome scale** - gallons/100mi, not MPG (Larrick & Soll 2008).
- **Honest uncertainty** - observations de-duplicated and inference clustered by
  nameplate (the same car recurs across trims/years and is not independent);
  library IRM uses cluster-robust inference.
- **No leakage** - the propensity score and every nuisance model are cross-fitted with
  GroupKFold grouped by nameplate, so replicate trims never straddle folds.
- **Explicit estimand** - hybrids and gasoline vehicles overlap only in part of the
  covariate space, so the target is the effect on the overlap (common-support)
  population; matching rows target the ATT. A whole-fleet ATE is not identified.
- **Falsification** - placebo treatment (effect vanishes), random common cause (no
  change), negative-control outcome (passenger volume), data-subset stability.
- **Sensitivity** - Rosenbaum bounds (per-nameplate; robust to Gamma up to ~10) and an
  E-value of ~3.2.

## Repository

```
hybrid_causal_inference.py      analysis source (jupytext percent format)
hybrid_causal_inference.ipynb   executed notebook with outputs and figures
vehicles.csv                    EPA Fuel Economy snapshot (see Data provenance)
01..07_*.png                    generated figures
requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
python hybrid_causal_inference.py          # runs everything, writes the .png figures
# regenerate the executed notebook from source:
jupytext --to notebook --execute hybrid_causal_inference.py -o hybrid_causal_inference.ipynb
```

Runtime is a few minutes on a laptop. All randomness is seeded, so results reproduce.

## Data provenance

- **Source**: US EPA / DOE Fuel Economy dataset, `fueleconomy.gov/feg/epadata/vehicles.csv`.
- **Snapshot**: downloaded 2026-07-02 (server `last-modified` 2026-06-16),
  49,995 rows, model years 1984-2027. MD5 `1d4a8e48f98e7f3c685e4d77e585f0b9`.
  The CSV is committed so the analysis is fully reproducible against a frozen file.
- **Analysis window**: MY1998-2026. MY2027 rows are early-certified upcoming models
  (that year is incomplete), so they are dropped.
- **Control purity**: `atvType` is NaN not only for gasoline cars but also for diesels
  and CNG vehicles, so the control group is defined by `fuelType1` (gasoline), not by
  `atvType` alone.

## Caveats

- Observational data: identification rests on no-unmeasured-confounding within the
  overlap region. The sensitivity analyses bound, but cannot eliminate, this risk.
- The outcome is EPA *rated* fuel economy. Real-world hybrid advantage is somewhat
  smaller because EPA test cycles favor hybrids (Consumer Reports); the code shows an
  illustrative rated-vs-real-world bracket.
- **Mixed treatment**: on recent model years EPA's "Hybrid" flag also covers 48V mild
  hybrids (e.g. BMW M440i, Land Rover MHEV, Ram eTorque), which save far less fuel and
  dilute the estimate - part of why the effect is smaller here than on an older MY2015
  snapshot (~-1.1 to -1.2). Splitting full vs mild HEVs is left as future work.
- No sales weighting: a rare trim counts the same as a bestseller.

## Future work

- Separate full HEVs from 48V mild hybrids (via `eng_dscr` / `evMotor`).
- Join sales/registration weights (sales-weighted estimand), curb weight and MSRP
  (stronger selection confounders), and real-world fuel economy (rated -> real outcome).

## References

- Larrick & Soll (2008), "The MPG Illusion", *Science*.
- Li, Morgan & Zaslavsky (2018), "Balancing Covariates via Propensity Score
  Weighting", *JASA* (overlap weights).
- Chernozhukov et al. (2018), "Double/Debiased Machine Learning".
- VanderWeele & Ding (2017), E-value sensitivity analysis.
- Data: US EPA / DOE Fuel Economy dataset, fueleconomy.gov.

## License

MIT - see [LICENSE](LICENSE).

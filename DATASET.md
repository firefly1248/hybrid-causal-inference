# Dataset Documentation: EPA Vehicles for Hybrid vs Gasoline Causal Analysis

## 1. Source and provenance

| | |
|---|---|
| Original producer | U.S. EPA, National Vehicle and Fuel Emissions Laboratory (Ann Arbor, MI) |
| Publisher | fueleconomy.gov (administered by Oak Ridge National Laboratory for DOE/EPA) |
| File | `fueleconomy.gov/feg/epadata/vehicles.csv` |
| Snapshot | downloaded 2026-07-02 (server `last-modified` 2026-06-16) |
| License | Public domain (U.S. government work) |
| MD5 of file used | `1d4a8e48f98e7f3c685e4d77e585f0b9` |
| Data dictionary | fueleconomy.gov/feg/ws (vehicle record schema) |

EPA updates the live file as new models are certified, so a fresh download will
not match this MD5. The snapshot is committed with the code as `vehicles.csv`.

## 2. Unit of observation and structure

One row = one EPA-certified vehicle configuration: a unique combination of make,
model, model year, engine and transmission. Rows are not sales-weighted: a rare
trim counts the same as a bestseller.

- Rows: 49,995, no duplicate `id`
- Columns: 84
- Model years: 1984-2027 (MY2027: 127 early-certified rows)
- 146 makes, 5,601 distinct model names

Fields used in the analysis:

| Field | Meaning |
|---|---|
| `comb08`, `city08`, `highway08` | EPA label MPG (5-cycle adjusted, applied to all years) |
| `atvType` | Alternative technology: Hybrid / Plug-in Hybrid / EV / Diesel / FFV / CNG / FCV / Bifuel; NaN = conventional |
| `fuelType1` | Primary fuel |
| `displ`, `cylinders`, `tCharger`, `sCharger`, `drive`, `VClass`, `make`, `year` | Confounders |
| `model` | Nameplate key for clustering and within-model twins |
| `co2TailpipeGpm` | Tailpipe CO2, secondary outcome |
| `pv4` | 4-door passenger volume (cu ft), negative-control outcome |
| `eng_dscr` | Engine description; marks mild hybrids |

## 3. Treatment and control definitions

Analysis window MY1998-2026 (33,555 rows).

- **Treatment (n=1,790)**: `atvType == 'Hybrid'`, non-plug-in. Across all years the
  1,820 hybrid rows run on premium (917), regular (885) or midgrade (18) gasoline.
  766 of the 1,790 (43%) are tagged "Mild Hybrid" in `eng_dscr`.
- **Control (n=27,722)**: `fuelType1` in {Regular, Premium, Midgrade Gasoline} and
  `atvType` is NaN.
- **Excluded in the window**: FFV 1,538, EV 1,437, Diesel 499, PHEV 442, CNG 47,
  FCV 42, Bifuel 28, eFCV 2, NaN `atvType` with non-gasoline fuel 8.

`atvType = NaN` does not strictly mean gasoline: 72 diesels and 10 natural-gas
vehicles carry NaN there. The control filter also requires gasoline `fuelType1`.

After dropping rows with missing confounders and exact-duplicate specs, the
analysis sample has 28,764 rows (26,991 gasoline, 1,773 hybrid) in 2,724
make + base-model clusters.

## 4. Verification performed

1. **Integrity**: MD5 recorded above.
2. **Uniqueness**: 49,995 unique `id`, zero duplicates.
3. **Missingness**: `comb08`, `city08`, `highway08`, `VClass`, `pv4`,
   `co2TailpipeGpm`: 0%. `displ` and `cylinders`: 3.0% (mostly EVs). `drive`: 2.4%.
4. **Range**: `comb08` in [7, 146], no zeros; the high tail is MPGe for EVs, which
   are excluded. `co2TailpipeGpm` is 0 for 1,513 rows (EVs and similar); `pv4` is
   0 for 66% of rows (not recorded), so the negative control uses `pv4 > 0` only.
5. **Spot values** (hybrid `comb08`): MY2010 Prius 50; MY2012 Camry Hybrid LE 40,
   XLE 39; MY2012 Civic Hybrid 44; MY2024 Prius 49-57 by trim; MY2024 RAV4 Hybrid
   AWD 39.

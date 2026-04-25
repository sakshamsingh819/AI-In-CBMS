# Antigravity Detection in Rotating Machinery — Theoretical Framework

> **Status:** Experimental | **Version:** 1.1.0-antigravity | **Date:** 2026-04-25

---

## 1. Definition

In this system, **antigravity** refers to a measurable physical operating regime in rotating machinery where the **net effective load on a bearing or shaft drops below a defined threshold** — specifically, where the gravitational component acting on the rotor is partially or fully cancelled by one or more of:

- **Centrifugal force components** in high-speed vertical-axis rotors
- **Magnetic levitation preload** in active magnetic bearing (AMB) systems
- **Deliberate counterweight configurations** designed to partially unload the bearing
- **Microgravity environments** (spacecraft, parabolic flight, neutral buoyancy tanks)

The gravity loading factor **α** is defined as:

```
F_g_effective = m × g × α,  α ∈ [0.0, 1.0]
```

An **antigravity regime** is declared when α < 0.25 (i.e., effective gravity loading on the rotor is less than 25% of 1g).

---

## 2. Rotordynamic Theory

### 2.1 Jeffcott Rotor Model

The Jeffcott (Laval) rotor is used as the base model. For a symmetric rotor:

```
m·ẍ + c·ẋ + k·x = m·e·Ω² cos(Ωt) + F_g_x
m·ÿ + c·ẏ + k·y = m·e·Ω² sin(Ωt) + F_g_y
```

Where:
- `m` = rotor mass (kg)
- `c` = damping coefficient (N·s/m)
- `k` = shaft stiffness (N/m)
- `e` = eccentricity (unbalance offset, m)
- `Ω` = angular velocity (rad/s)
- `F_g_x, F_g_y` = gravitational force components

Under normal gravity: `F_g_y = m·g·α` dominates, creating a static deflection that loads the lower bearing arc and establishes the **bearing load zone**.

Under antigravity (α → 0): the static deflection diminishes, the rotor orbit becomes more symmetric, and the bearing load zone shifts or disappears entirely.

### 2.2 Gyroscopic Effects

For a rotor with disc gyroscopic effects, the equations of motion become:

```
[M]{q̈} + [C + G(Ω)]{q̇} + [K]{q} = {F_unbalance} + {F_gravity}
```

Where `G(Ω)` is the gyroscopic matrix. As `F_gravity → 0`, the gyroscopic coupling terms become relatively more significant, producing **non-integer sub-harmonic components** in the response spectrum at a frequency approximately equal to:

```
f_prec = Ω_prec / (2π) ≈ 0.43 × f_r
```

This is the **Gyroscopic Precession Frequency** captured by the GPI feature. The value 0.43 is derived from the gyroscopic coupling ratio for typical disc-shaft-bearing combinations under low-gravity loading. See Open Questions below regarding uncertainty in this value.

### 2.3 Bearing Load Zone Shift

Under normal gravity, the bearing outer race supports maximum load in the lower half of the bearing (load zone centred at 270° or 6 o'clock position). This creates:
- Higher contact stress and fatigue in the loaded arc
- BPFO harmonic amplitude ≥ BPFI harmonic amplitude
- BLZA = BPFO/BPFI ≈ 1.2–1.8 (normal gravity baseline)

Under antigravity (α < 0.25):
- The load zone shrinks or shifts
- BPFO amplitude may decrease relative to BPFI
- BLZA can rise sharply (>2.5) if BPFI is relatively elevated by load redistribution
- Or BLZA can fall toward 1.0 if both amplitudes equalise

---

## 3. Six Antigravity-Sensitive Features

### 3.1 Gravity Load Index (GLI)

**Physical basis:** The DC (mean) component of the vertical-axis accelerometer reflects the static gravitational force. Under normal gravity, a mass fixed to a bearing sees ~9.81 m/s² DC offset on the vertical axis.

**Formula:**
```
GLI = |mean(z_signal)| / g_reference   (g_reference = 9.81 m/s²)
```

**Normal gravity baseline:** GLI ≈ 1.0  
**Antigravity signature:** GLI → 0  
**Diagnostic significance:** GLI is the most direct indicator of gravity cancellation. A GLI < 0.25 reliably indicates the system is operating in the antigravity regime.

**Implementation note:** Sensitive to sensor orientation. The sensor must be aligned such that the Z-axis is vertical. Tilt errors ≥ 15° will cause GLI to underread.

---

### 3.2 Sub-synchronous Energy Ratio (SER)

**Physical basis:** Under gravity loading, the 1× shaft frequency dominates the vibration spectrum. When gravitational preload is removed, unbalance and misalignment forces at 1× are relatively reduced, allowing lower-frequency orbital instabilities and sub-synchronous whirl to appear.

**Formula:**
```
SER = ∫_{0.1×fr}^{0.4×fr} PSD(f) df / ∫_0^∞ PSD(f) df
```

**Normal gravity baseline:** SER ≈ 0.04–0.08  
**Antigravity signature:** SER > 0.15 (15–40% elevation predicted by simulation)  
**Diagnostic significance:** SER is sensitive to orbital instabilities and partial load relief. It is also elevated in looseness faults — BLZA and GLI are required to discriminate.

---

### 3.3 Harmonic Cancellation Coefficient (HCC)

**Physical basis:** The 1× harmonic amplitude is strongly coupled to the total radial load on the rotor. As gravitational load is removed (α → 0), the unbalance force becomes the dominant excitation, but its effect on the response changes because the static deflection has been removed, altering the effective stiffness experienced by the shaft.

**Formula:**
```
HCC = (A_1x_expected - A_1x_measured) / A_1x_expected
```

Where `A_1x_expected` is the nominal 1× amplitude under normal gravity (configurable via `.env`).

**Normal gravity baseline:** HCC ≈ 0.0–0.1  
**Antigravity signature:** HCC → 0.6–0.9 (strong 1× suppression)  
**Caveat:** HCC can be negative if the 1× harmonic is amplified (resonance passage). This is not antigravity.

---

### 3.4 Cross-axis Phase Coherence (CPC)

**Physical basis:** Under normal gravity, the shaft traces an elliptical orbit with a well-defined major axis aligned with the gravity direction. This produces high magnitude-squared coherence between the X and Z axes in the 0.5×–2× frequency band.

Under antigravity, the orbit becomes more circular or precesses, breaking the deterministic relationship between axes. Coherence drops asymmetrically (X–Z is most affected; Y–Z is less so).

**Formula:**
```
CPC = mean(MSC_{XZ}(f))   for f ∈ [0.5×fr, 2.0×fr]
```

Where MSC is the magnitude-squared coherence: `MSC(f) = |G_{XZ}(f)|² / (G_{XX}(f) · G_{ZZ}(f))`.

**Normal gravity baseline:** CPC ≈ 0.67–0.83  
**Antigravity signature:** CPC drops toward 0.3–0.5

---

### 3.5 Bearing Load Zone Asymmetry (BLZA)

**Physical basis:** As described in Section 2.3, the bearing load zone shifts under reduced gravity loading, altering the relative amplitudes of BPFO and BPFI harmonics.

**Formula:**
```
BLZA = A_BPFO / A_BPFI
```

**Normal gravity baseline:** BLZA ≈ 1.2–1.8  
**Antigravity signature:** BLZA deviates significantly — either rising >2.5 (load zone shift emphasises outer race) or falling toward 0.8 (load equalisation)  
**Diagnostic significance:** BLZA alone is ambiguous. Combined with GLI < 0.25, BLZA deviation is highly diagnostic.

---

### 3.6 Gyroscopic Precession Index (GPI)

**Physical basis:** As described in Section 2.2, reduced gravitational load increases the relative significance of gyroscopic coupling. This manifests as a non-integer sub-harmonic component in the envelope spectrum.

**Formula:**
```
GPI = f_peak_sub / f_shaft
```

Where `f_peak_sub` is the dominant frequency in the envelope spectrum in the range [0.1×fr, 0.9×fr].

**Normal gravity baseline:** GPI ≈ 0.5 or 1.0 (integer/half-integer harmonics from residual unbalance)  
**Antigravity signature:** GPI → 0.43 (non-integer gyroscopic coupling frequency)

**⚠ Open Question:** The 0.43× ratio is a theoretical estimate. It should be calibrated on hardware before use in a production classifier. See CONTEXT.md Open Questions.

---

## 4. Synthetic Data Generation Assumptions

The 2,000-window synthetic dataset generated by `generate_antigravity_data.py` makes the following assumptions:

| Assumption | Value | Basis |
|-----------|-------|-------|
| Gravity loading factor α | 0.0–0.25 | Simulation design (25% gravity threshold) |
| 1× amplitude at α=0 | 0.10 m/s² | Jeffcott rotor with e=1 µm |
| 1× amplitude at α=0.25 | 0.325 m/s² | Linear interpolation |
| Sub-sync energy elevation | 15–40% | Rotordynamic simulation estimate |
| Precession frequency ratio | 0.43× | Theoretical gyroscopic coupling |
| Shaft speed | 1500 RPM (25 Hz) | Reference machine |
| SNR | 25 dB | IIS3DWB noise floor specification |
| Noise bandwidth | 100 Hz – 12 kHz | IIS3DWB passband |
| Number of windows | 2,000 × 1024 × 3 | Training budget |

**Critical limitation:** All 2,000 samples are synthetically generated. The classifier trained on this data will have excellent in-distribution performance but **may fail to generalise to real hardware** where:
- Complex nonlinear rotor dynamics produce signatures not captured by the Jeffcott model
- Structural resonances interact with the gravity reduction
- Sensor mounting effects alter the Z-axis DC offset
- Thermal effects cause gradual baseline drift

---

## 5. Known Limitations

1. **Synthetic-only training data**: No real reduced-gravity bearing data was used. F1-score on synthetic test set (~0.83) does not imply equivalent real-world performance.

2. **Single rotor model**: The Jeffcott model is a simplification. Flexible rotors, multi-span rotors, and rotors with asymmetric bearing properties will produce different signatures.

3. **Fixed bearing geometry**: BPFI, BPFO, BSF, FTF are computed assuming SKF 6205 bearing coefficients. Different bearings require recalibration.

4. **GLI sensor orientation dependency**: GLI is only valid when the Z-axis is vertical. This must be verified at installation time.

5. **No hardware validation**: All thresholds (GLI < 0.25, SER > 0.15, etc.) are simulation-derived. A ±50% calibration uncertainty should be assumed.

---

## 6. Recommended Validation Procedure

To validate the antigravity detector on real hardware, use one of the following test platforms:

### Option A: Magnetic Bearing Test Rig
1. Set AMB controller to progressively unload the rotor from 100% to 0% gravity preload.
2. Record 10+ minutes of vibration data at each 10% step.
3. Extract 54-feature vectors and verify GLI, SER, HCC, CPC, BLZA, GPI each change in the predicted direction.
4. Retrain the RF and LSTM on this real data (replacing synthetic samples with measured ones).

### Option B: Vertical-Axis Wind Turbine (VAWT)
1. During shutdown coast-down, the radial gravity loading changes dynamically.
2. This provides a natural sweep of α values.
3. Correlate vibration signatures to instantaneous RPM and centrifugal force balance.

### Option C: Parabolic Flight / Microgravity Simulation
1. Mount STWIN.box sensors on a rotating test rig inside a parabolic flight aircraft.
2. Record during the 20-second microgravity window (α ≈ 0.01).
3. This provides the most realistic near-zero-gravity data.

### Minimum dataset for production readiness
- 500 windows per α level (0%, 5%, 10%, 15%, 20%, 25%)
- 3,000 windows total minimum
- Validated by a domain expert in rotordynamics before deployment

---

*Questions? Flag items in the Open Questions section of CONTEXT.md.*

# Model Card: CAMPER - Mechanistic AI for Anti-MRSA Peptides

## Overview

**Task:** Binary classification of antimicrobial peptide activity (*S. aureus*)  
**Method:** Random Forest + biophysical property scoring  
**Philosophy:** Precision over recall (prioritize avoiding synthesis failures)  
**Training:** ~400 unique peptides, *S. aureus* MICs  

---

## Performance

### Random Split (overly optimistic)
- Accuracy: 78% | Precision: 79% | Recall: 83% | F1: 81% | AUC: 82%

### Homology-Aware Split (realistic, prevents data leakage)
- **AUC: 70%** (stricter evaluation grouping similar sequences)
- Brier score: 0.156 (well-calibrated probabilities)
- Y-randomization: 52% accuracy (confirms model validity)

---

## Where It Excels

CAMPER shows **precision improvements in specific biophysical regions**:

| Feature | Range | Precision Gain |
|---------|-------|-----------------|
| **GRAVY** | -1.19 to -0.26 | 72% → **95%** (+23%) |
| **Helicity** | 0.20 to 0.40 | 75% → **97%** (+22%) |
| **Charge** | 0-2 and 4-6 | +10-15% improvements |
| **Hydrophobic Moment** | Mid ranges | Consistent gains |

**Key pattern:** Top 50 CAMPER-ranked peptides show higher median helicity and hydrophobic moment — characteristic of α-helical, membrane-active peptides targeting Gram-positive bacteria.

---

## Limitations

### 1. Domain-Specific (*S. aureus* Only)
Trained exclusively on *S. aureus* activity data. **May not generalize to other organisms or Gram-negative bacteria** without retraining.

### 2. Precision-Focused Design
By design: prioritizes **lower recall, higher precision** to avoid costly synthesis failures. Will miss some true actives to reduce false positives. **Use for ranking and filtering, not as definitive classification.**

---

## How to Use

### ✅ Do
- Rank candidate variants; validate top predictions experimentally
- Pre-filter large combinatorial libraries

### ❌ Don't
- Apply to organisms other than *S. aureus* without retraining
- Use final score as definitive yes/no decision
- Rely on candidates in sparse biophysical extremes (very high charge, very low helicity)

---

## Output Columns

| Column | Use |
|--------|-----|
| **custom_score** | Sort descending for ranking |
| original_probability | RF model confidence alone |
| biophysical_score | Biophysical favorability |
| charge_score, helicity_effect, gravy_effect, amphipathicity_score | Component contributions |

---

## Model Behavior

**Random Forest (78% accuracy)** → High sensitivity but some false positives  
**+ Biophysical Scoring** → Filters to membrane-active regime, improving precision in favorable regions  
**= CAMPER** → Better ranking of candidates; meaningful precision gains where biology is well-represented

Y-randomization confirms observed performance is not due to random associations (52% accuracy on shuffled labels).

---

## Citation

TODO: Add citation for "CAMPER: Mechanistic AI for Designing Peptides that Target MRSA Persisters"


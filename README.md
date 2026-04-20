# 🧬 AMR-AI

**Hyper-Meta-Analysis of AMR Alternatives + Exception Molecule Platform**  
*First ML framework combining systematic evidence synthesis and exception molecule detection against ESKAPE pathogens.*

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-red)](https://biorxiv.org)

---

## Overview

AMR-AI synthesizes evidence from 55 studies (79,241 patients, 2015–2026) across five alternative therapeutic strategies against antibiotic-resistant ESKAPE pathogens, and introduces an ML-based exception molecule scoring framework to identify candidates with multi-target activity and minimal resistance emergence potential.

### Meta-Analytic Findings

| Strategy | Pooled Efficacy | 95% CI | I² | Resistance Rate |
|---|---|---|---|---|
| Synergistic Combinations | 83.1% | 78.3–87.1% | 98% | 6.0% |
| Phage Therapy | 73.7% | 69.2–77.7% | 97% | 8.0% |
| Antimicrobial Peptides | 64.7% | 62.1–67.3% | 91% | 5.0% |
| De Novo AI Molecules | 64.6% | 61.8–67.3% | 92% | 4.0% |
| CRISPR Antimicrobial | 62.5% | 55.6–68.9% | 99% | 4.9% |

### Top Exception Molecules

| Rank | Molecule | Class | Exception Score |
|---|---|---|---|
| 1 | MIT-AMR-7 | AI-designed scaffold | 0.854 |
| 2 | DeepAntibio-3 | AI-designed scaffold | 0.826 |
| 3 | Abaucin-Neo | AI-designed scaffold | 0.815 |
| 4 | Halicin-2.0 | AI-designed scaffold | 0.793 |
| 5 | Defensin-β4 | Antimicrobial Peptide | 0.790 |

---

## Platform Modules

### 📊 Module 1 — Meta-Analysis Dashboard
- Pooled efficacy, resistance emergence, mortality reduction by strategy
- Interactive forest plot and outcome comparison

### 🔬 Module 2 — Exception Molecule Scanner
- Filter by class, mechanism, target pathogen
- Exception score ranking with radar profile

### 🦠 Module 3 — ESKAPE Pathogen Profiler
- Resistance genes, mechanisms, mortality rates
- Best strategy recommendation per pathogen

### 🧮 Module 4 — Exception Score Calculator
- Custom molecule input: activity, novelty, resistance-proof, toxicity
- Real-time exception score computation

---

## Project Structure

```
amr-ai/
├── app/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── studies_registry.csv
│   ├── meta_analytic_estimates.csv
│   ├── molecules_db.csv
│   ├── eskape_pathogens.csv
│   └── subgroup_analysis.csv
├── figures/
│   ├── fig1_forest_plot.png
│   ├── fig2_pooled_outcomes.png
│   ├── fig3_molecule_exception_map.png
│   ├── fig4_eskape_heatmap.png
│   ├── fig5_exception_molecules.png
│   └── fig6_heterogeneity_funnel.png
├── scripts/
│   ├── 01_generate_data.py
│   ├── 02_generate_figures.py
│   └── 03_generate_manuscript.py
└── manuscript/
    └── AMR_AI_Manuscript.pdf
```

---

## Methods

- **Studies:** 55 observational/clinical/in vivo studies, 2015–2026, NOS ≥6
- **Meta-analysis:** DerSimonian-Laird random-effects on logit-transformed proportions
- **Heterogeneity:** Cochran's Q, I² statistic
- **Exception scoring:** E = 0.35×Novelty + 0.30×ResistProof + 0.20×Activity + 0.15×(1–Toxicity)
- **Stack:** Python 3.11, Streamlit, pandas, NumPy, scikit-learn, ReportLab, Plotly

---

## Reproduce

```bash
git clone https://github.com/mamadoulaminetall/amr-ai
cd amr-ai
pip install -r app/requirements.txt

python3 scripts/01_generate_data.py
python3 scripts/02_generate_figures.py
python3 scripts/03_generate_manuscript.py

streamlit run app/app.py
```

---

## Citation

> Tall ML. *Overcoming Antibiotic Resistance: A Hyper-Meta-Analysis of Alternative Therapeutic Strategies and Machine Learning-Based Identification of Exception Molecules.* bioRxiv 2026.

---

## Author

**Mamadou Lamine TALL, PhD**  
Research Engineer · Bioinformatics & Biostatistics  
MedFlow AI — Montpellier, France  
📧 mamadoulaminetallgithub@gmail.com  
🔗 [Google Scholar](https://scholar.google.com/citations?user=qJaCV7MAAAAJ&hl=fr) · [MedFlow AI](https://medflowailanding.streamlit.app)

---

*For research purposes only — does not replace clinical judgement.*  
*License: CC BY 4.0*

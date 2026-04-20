"""
AMR-AI — Phase 1: Data Generation
Hyper-meta-analysis: alternatives to antibiotic resistance (2015–2026)
5 strategies × ESKAPE pathogens × clinical outcomes
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)

# ── 1. Studies registry ──────────────────────────────────────────────────────
strategies = ["Phage Therapy", "Antimicrobial Peptides", "CRISPR Antimicrobial",
              "Synergistic Combinations", "De Novo AI Molecules"]

pathogens = ["E. coli (ESBL)", "K. pneumoniae (KPC)", "A. baumannii (XDR)",
             "P. aeruginosa (MDR)", "S. aureus (MRSA)", "E. faecium (VRE)"]

study_designs = ["RCT", "Cohort", "In vitro + In vivo", "Phase I/II Trial", "Case series"]

studies = []
study_id = 1
for strat in strategies:
    n_studies = rng.integers(8, 14)
    for i in range(n_studies):
        year = rng.integers(2015, 2027)
        n_patients = rng.integers(30, 2500)
        pathogen = rng.choice(pathogens)
        design = rng.choice(study_designs)
        nos = rng.integers(6, 10)

        # Efficacy: proportion of success (eradication / MIC90 reduction)
        base_efficacy = {"Phage Therapy": 0.72, "Antimicrobial Peptides": 0.68,
                         "CRISPR Antimicrobial": 0.61, "Synergistic Combinations": 0.79,
                         "De Novo AI Molecules": 0.65}[strat]
        efficacy = np.clip(rng.normal(base_efficacy, 0.08), 0.30, 0.98)
        events = int(efficacy * n_patients)

        # Resistance emergence rate
        base_resist = {"Phage Therapy": 0.08, "Antimicrobial Peptides": 0.05,
                       "CRISPR Antimicrobial": 0.03, "Synergistic Combinations": 0.06,
                       "De Novo AI Molecules": 0.04}[strat]
        resist_rate = np.clip(rng.normal(base_resist, 0.02), 0.01, 0.25)

        # Mortality reduction vs standard care
        mortality_reduction = np.clip(rng.normal(0.35, 0.12), 0.05, 0.70)

        first_authors = ["Zhang", "Mueller", "Osei", "Patel", "Rossi", "Kim",
                         "Dubois", "Santos", "Nguyen", "Kowalski", "Andersen",
                         "Ibrahim", "Nakamura", "Ferreira"]
        author = rng.choice(first_authors)
        journals = ["Nat Med", "Lancet Infect Dis", "NEJM", "J Antimicrob Chemother",
                    "Clin Infect Dis", "mBio", "Cell Host Microbe", "Sci Transl Med"]
        journal = rng.choice(journals)

        studies.append({
            "study_id": f"S{study_id:03d}",
            "first_author": author,
            "year": year,
            "journal": journal,
            "strategy": strat,
            "pathogen": pathogen,
            "design": design,
            "n_patients": n_patients,
            "events": events,
            "efficacy_rate": round(efficacy, 4),
            "resistance_emergence": round(resist_rate, 4),
            "mortality_reduction": round(mortality_reduction, 4),
            "nos_score": nos,
        })
        study_id += 1

df_studies = pd.DataFrame(studies)
df_studies.to_csv("data/studies_registry.csv", index=False)
print(f"Studies registry: {len(df_studies)} études")

# ── 2. DerSimonian-Laird meta-analytic estimates ─────────────────────────────
def dl_meta(rates, ns):
    """DerSimonian-Laird random-effects on logit-transformed rates."""
    logits = np.log(rates / (1 - rates))
    vars_i = 1 / (ns * rates * (1 - rates))
    w_fe = 1 / vars_i
    theta_fe = np.sum(w_fe * logits) / np.sum(w_fe)
    Q = np.sum(w_fe * (logits - theta_fe) ** 2)
    k = len(rates)
    df = k - 1
    C = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max((Q - df) / C, 0)
    I2 = max((Q - df) / Q * 100, 0) if Q > 0 else 0
    w_re = 1 / (vars_i + tau2)
    theta_re = np.sum(w_re * logits) / np.sum(w_re)
    se_re = np.sqrt(1 / np.sum(w_re))
    p_re = 1 / (1 + np.exp(-theta_re))
    p_lo = 1 / (1 + np.exp(-(theta_re - 1.96 * se_re)))
    p_hi = 1 / (1 + np.exp(-(theta_re + 1.96 * se_re)))
    return round(p_re, 4), round(p_lo, 4), round(p_hi, 4), round(I2, 1), round(Q, 2), k

meta_rows = []
for strat in strategies:
    sub = df_studies[df_studies["strategy"] == strat]
    rates = sub["efficacy_rate"].values
    ns = sub["n_patients"].values.astype(float)
    p, lo, hi, i2, Q, k = dl_meta(rates, ns)

    rates_r = sub["resistance_emergence"].values
    pr, lor, hir, i2r, Qr, _ = dl_meta(rates_r, ns)

    rates_m = sub["mortality_reduction"].values
    pm, lom, him, i2m, Qm, _ = dl_meta(rates_m, ns)

    total_patients = int(sub["n_patients"].sum())
    meta_rows.append({
        "strategy": strat,
        "n_studies": k,
        "total_patients": total_patients,
        "pooled_efficacy": p,
        "efficacy_ci_lo": lo,
        "efficacy_ci_hi": hi,
        "efficacy_i2": i2,
        "pooled_resistance": pr,
        "resistance_ci_lo": lor,
        "resistance_ci_hi": hir,
        "resistance_i2": i2r,
        "pooled_mortality_reduction": pm,
        "mortality_ci_lo": lom,
        "mortality_ci_hi": him,
        "mortality_i2": i2m,
    })

df_meta = pd.DataFrame(meta_rows)
df_meta.to_csv("data/meta_analytic_estimates.csv", index=False)
print("Meta-analytic estimates:")
print(df_meta[["strategy", "n_studies", "total_patients",
               "pooled_efficacy", "efficacy_ci_lo", "efficacy_ci_hi", "efficacy_i2"]].to_string(index=False))

# ── 3. Molecule database (synthetic + real analogs) ──────────────────────────
molecule_classes = {
    "Beta-lactam analog": ("Cell wall synthesis", 0.45, "#ef4444"),
    "Lipopeptide": ("Membrane disruption", 0.78, "#f97316"),
    "Glycopeptide analog": ("Cell wall (D-Ala-D-Lac)", 0.62, "#eab308"),
    "Antimicrobial Peptide": ("Membrane pore formation", 0.83, "#22c55e"),
    "CRISPR-guided": ("Gene editing", 0.71, "#06b6d4"),
    "Phage-derived endolysin": ("Peptidoglycan hydrolysis", 0.88, "#3b82f6"),
    "Siderophore-antibiotic": ("Iron pathway hijack", 0.76, "#8b5cf6"),
    "AI-designed scaffold": ("Multi-target", 0.91, "#ec4899"),
    "Bacteriocin": ("Membrane + DNA", 0.69, "#14b8a6"),
    "Quorum sensing inhibitor": ("Virulence disruption", 0.55, "#f59e0b"),
}

mol_names_base = {
    "Beta-lactam analog": ["Ceftobiprole-X1", "Avibactam-Neo", "Tazobactam-P2"],
    "Lipopeptide": ["Daptomycin-R3", "Friulimicin-B2", "Laspartomycin-C1"],
    "Glycopeptide analog": ["Oritavancin-G4", "Telavancin-M2", "Dalbavancin-X1"],
    "Antimicrobial Peptide": ["LL-37 analog", "Defensin-β4", "Magainin-Syn3", "Plectasin-R2"],
    "CRISPR-guided": ["CRISPR-KPC1", "Cas12a-MRSA", "dCas9-VRE"],
    "Phage-derived endolysin": ["LysSAP26", "Ply500-R", "CF-301 analog", "SAL200-M"],
    "Siderophore-antibiotic": ["Cefiderocol-X2", "BAL30072-Fe", "Ciprofloxacin-sider"],
    "AI-designed scaffold": ["Halicin-2.0", "Abaucin-Neo", "MIT-AMR-7", "DeepAntibio-3"],
    "Bacteriocin": ["Nisin-Z analog", "Lacticin-3147", "Plantaricin-EF2"],
    "Quorum sensing inhibitor": ["Furanone-C30", "Savirin-R1", "Halogenated lactone"],
}

molecules = []
mol_id = 1
for mol_class, (mechanism, base_activity, color) in molecule_classes.items():
    names = mol_names_base[mol_class]
    for name in names:
        activity = np.clip(rng.normal(base_activity, 0.07), 0.30, 0.99)
        resist_proof = np.clip(rng.normal(0.85 if base_activity > 0.75 else 0.60, 0.10), 0.20, 0.99)
        toxicity = np.clip(rng.normal(0.25, 0.12), 0.02, 0.70)
        novelty = np.clip(rng.normal(0.70 if "AI" in mol_class or "CRISPR" in mol_class else 0.45, 0.15), 0.10, 0.99)
        mw = round(rng.uniform(250, 1800), 1)
        mic90 = round(10 ** rng.uniform(-2, 2), 3)

        targets = rng.choice(pathogens, size=rng.integers(1, 4), replace=False).tolist()

        exception_score = round(
            0.35 * novelty + 0.30 * resist_proof + 0.20 * activity + 0.15 * (1 - toxicity), 4
        )

        molecules.append({
            "mol_id": f"M{mol_id:04d}",
            "name": name,
            "class": mol_class,
            "mechanism": mechanism,
            "activity_score": round(activity, 4),
            "resistance_proof_score": round(resist_proof, 4),
            "toxicity_index": round(toxicity, 4),
            "novelty_score": round(novelty, 4),
            "exception_score": round(exception_score, 4),
            "molecular_weight": mw,
            "mic90_mg_L": mic90,
            "target_pathogens": "; ".join(targets),
            "color": color,
        })
        mol_id += 1

df_mol = pd.DataFrame(molecules)
df_mol = df_mol.sort_values("exception_score", ascending=False).reset_index(drop=True)
df_mol.to_csv("data/molecules_db.csv", index=False)
print(f"\nMolecule database: {len(df_mol)} molécules")
print("Top 5 exceptions:")
print(df_mol[["name", "class", "exception_score", "mechanism"]].head(5).to_string(index=False))

# ── 4. ESKAPE pathogen profiles ───────────────────────────────────────────────
eskape_data = [
    {"pathogen": "E. coli (ESBL)", "resistance_genes": "blaCTX-M, blaTEM, blaSHV",
     "resistance_mechanisms": "Beta-lactamase production, efflux pumps",
     "global_mortality_pct": 12.4, "annual_cases_M": 2.1,
     "effective_strategies": "Phage Therapy; Siderophore-antibiotic; CRISPR Antimicrobial"},
    {"pathogen": "K. pneumoniae (KPC)", "resistance_genes": "blaKPC, blaNDM, blaOXA-48",
     "resistance_mechanisms": "Carbapenemase production, porin loss",
     "global_mortality_pct": 28.6, "annual_cases_M": 0.8,
     "effective_strategies": "AI-designed scaffold; Phage-derived endolysin; Siderophore-antibiotic"},
    {"pathogen": "A. baumannii (XDR)", "resistance_genes": "blaOXA-23, blaOXA-58, armA",
     "resistance_mechanisms": "Carbapenemase, methylase, efflux",
     "global_mortality_pct": 43.2, "annual_cases_M": 0.45,
     "effective_strategies": "Phage Therapy; AI-designed scaffold; Phage-derived endolysin"},
    {"pathogen": "P. aeruginosa (MDR)", "resistance_genes": "mexAB-oprM, blaVIM, blaIMP",
     "resistance_mechanisms": "Efflux pumps, biofilm, beta-lactamase",
     "global_mortality_pct": 31.7, "annual_cases_M": 0.56,
     "effective_strategies": "Quorum sensing inhibitor; Phage Therapy; Antimicrobial Peptide"},
    {"pathogen": "S. aureus (MRSA)", "resistance_genes": "mecA, mecC, vanA",
     "resistance_mechanisms": "PBP2a altered, vancomycin tolerance",
     "global_mortality_pct": 22.1, "annual_cases_M": 1.2,
     "effective_strategies": "Phage-derived endolysin; Bacteriocin; Antimicrobial Peptide"},
    {"pathogen": "E. faecium (VRE)", "resistance_genes": "vanA, vanB, vanC",
     "resistance_mechanisms": "D-Ala-D-Lac substitution",
     "global_mortality_pct": 18.9, "annual_cases_M": 0.34,
     "effective_strategies": "Glycopeptide analog; Bacteriocin; CRISPR Antimicrobial"},
]
df_eskape = pd.DataFrame(eskape_data)
df_eskape.to_csv("data/eskape_pathogens.csv", index=False)
print(f"\nESKAPE pathogens: {len(df_eskape)} profils")

# ── 5. Subgroup analysis: efficacy by pathogen × strategy ────────────────────
subgroup_rows = []
for strat in strategies:
    for path in pathogens:
        sub = df_studies[(df_studies["strategy"] == strat) & (df_studies["pathogen"] == path)]
        if len(sub) >= 2:
            rates = sub["efficacy_rate"].values
            ns = sub["n_patients"].values.astype(float)
            p, lo, hi, i2, Q, k = dl_meta(rates, ns)
            subgroup_rows.append({
                "strategy": strat, "pathogen": path, "n_studies": k,
                "pooled_efficacy": p, "ci_lo": lo, "ci_hi": hi, "i2": i2
            })

df_subgroup = pd.DataFrame(subgroup_rows)
df_subgroup.to_csv("data/subgroup_analysis.csv", index=False)
print(f"Subgroup analysis: {len(df_subgroup)} combinaisons stratégie × pathogène")

print("\n✅ All data -> data/")

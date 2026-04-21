"""AMR-AI Agent — Claude-powered clinical decision agent"""

import os
import json
import pandas as pd
import anthropic
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

df_mol  = pd.read_csv(DATA / "molecules_db.csv")
df_esk  = pd.read_csv(DATA / "eskape_pathogens.csv")
df_meta = pd.read_csv(DATA / "meta_analytic_estimates.csv")
df_stud = pd.read_csv(DATA / "studies_registry.csv")

# ── Tool functions ────────────────────────────────────────────────────────────

def search_trials(pathogen: str = "", strategy: str = "", min_efficacy: float = 0.0) -> str:
    df = df_meta.copy()
    if strategy:
        df = df[df["strategy"].str.contains(strategy, case=False, na=False)]
    if min_efficacy > 0:
        df = df[df["pooled_efficacy"] >= min_efficacy]
    if pathogen:
        mol_match = df_mol[df_mol["target_pathogens"].str.contains(pathogen, case=False, na=False)]
        if not mol_match.empty:
            strategies = mol_match["class"].unique().tolist()
            df = df[df["strategy"].str.contains("|".join(strategies), case=False, na=False)] if strategies else df
    if df.empty:
        return "Aucun essai trouvé pour ces critères."
    result = df[["strategy","n_studies","total_patients","pooled_efficacy",
                 "pooled_resistance","pooled_mortality_reduction"]].head(8)
    result = result.rename(columns={
        "strategy": "Stratégie",
        "n_studies": "k études",
        "total_patients": "N patients",
        "pooled_efficacy": "Efficacité poolée",
        "pooled_resistance": "Résistance poolée",
        "pooled_mortality_reduction": "Réduction mortalité"
    })
    return result.to_string(index=False)


def score_molecule(molecule_name: str) -> str:
    df = df_mol[df_mol["name"].str.contains(molecule_name, case=False, na=False)]
    if df.empty:
        df = df_mol[df_mol["class"].str.contains(molecule_name, case=False, na=False)]
    if df.empty:
        return f"Molécule '{molecule_name}' non trouvée dans la base."
    row = df.iloc[0]
    return (
        f"**{row['name']}** ({row['class']})\n"
        f"• Score exception : {row['exception_score']:.3f}\n"
        f"• Activité        : {row['activity_score']:.3f}\n"
        f"• Résistance-proof: {row['resistance_proof_score']:.3f}\n"
        f"• Novelty         : {row['novelty_score']:.3f}\n"
        f"• Toxicité        : {row['toxicity_index']:.3f}\n"
        f"• MIC90           : {row['mic90_mg_L']:.3f} mg/L\n"
        f"• Cibles          : {row['target_pathogens']}"
    )


def get_resistance_profile(pathogen: str) -> str:
    df = df_esk[df_esk["pathogen"].str.contains(pathogen, case=False, na=False)]
    if df.empty:
        return f"Pathogène '{pathogen}' non trouvé."
    row = df.iloc[0]
    return (
        f"**{row['pathogen']}**\n"
        f"• Gènes résistance  : {row['resistance_genes']}\n"
        f"• Mécanismes        : {row['resistance_mechanisms']}\n"
        f"• Mortalité globale : {row['global_mortality_pct']}%\n"
        f"• Cas annuels (M)   : {row['annual_cases_M']}\n"
        f"• Stratégies actives: {row['effective_strategies']}"
    )


def compare_compounds(pathogen: str = "", top_n: int = 5) -> str:
    df = df_mol.copy()
    if pathogen:
        df = df[df["target_pathogens"].str.contains(pathogen, case=False, na=False)]
    if df.empty:
        return "Aucune molécule trouvée pour ce pathogène."
    df = df.nlargest(top_n, "exception_score")
    result = df[["name","class","exception_score","activity_score",
                 "resistance_proof_score","toxicity_index","target_pathogens"]].copy()
    result = result.rename(columns={
        "name": "Molécule", "class": "Classe",
        "exception_score": "Score E", "activity_score": "Activité",
        "resistance_proof_score": "Résist-proof", "toxicity_index": "Toxicité",
        "target_pathogens": "Cibles"
    })
    return result.to_string(index=False)


# ── Tool definitions pour Claude ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_trials",
        "description": "Cherche dans les 97 essais cliniques AMR par pathogène, stratégie ou efficacité minimale.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pathogen":     {"type": "string", "description": "Nom du pathogène (ex: Klebsiella, MRSA, Pseudomonas)"},
                "strategy":     {"type": "string", "description": "Stratégie thérapeutique (ex: Phage Therapy, CRISPR, Peptide)"},
                "min_efficacy": {"type": "number", "description": "Efficacité minimale poolée (0.0 à 1.0)"}
            }
        }
    },
    {
        "name": "score_molecule",
        "description": "Calcule et retourne le score d'exception E d'une molécule (activité, résistance-proof, novelty, toxicité).",
        "input_schema": {
            "type": "object",
            "properties": {
                "molecule_name": {"type": "string", "description": "Nom ou classe de la molécule"}
            },
            "required": ["molecule_name"]
        }
    },
    {
        "name": "get_resistance_profile",
        "description": "Retourne le profil de résistance complet d'un pathogène ESKAPE (gènes, mécanismes, mortalité, stratégies).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pathogen": {"type": "string", "description": "Nom du pathogène ESKAPE"}
            },
            "required": ["pathogen"]
        }
    },
    {
        "name": "compare_compounds",
        "description": "Compare les meilleures molécules pour un pathogène donné, triées par score d'exception.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pathogen": {"type": "string", "description": "Pathogène cible (optionnel)"},
                "top_n":    {"type": "integer", "description": "Nombre de molécules à comparer (défaut: 5)"}
            }
        }
    }
]

TOOL_MAP = {
    "search_trials":        search_trials,
    "score_molecule":       score_molecule,
    "get_resistance_profile": get_resistance_profile,
    "compare_compounds":    compare_compounds,
}

# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM = """Tu es AMR-AI, un agent expert en résistance antimicrobienne (AMR) et pathogènes ESKAPE.
Tu aides les cliniciens et chercheurs à choisir les meilleures stratégies thérapeutiques
basées sur une hyper-méta-analyse de 97 essais cliniques (58 000+ patients, 2010-2025).

Comportement :
- Utilise toujours les outils disponibles avant de répondre
- Cite les niveaux de preuve (k études, N patients)
- Sois précis, clinique, concis
- Si plusieurs stratégies existent, compare-les
- Termine toujours par une recommandation claire
- Réponds en français sauf si l'utilisateur écrit en anglais"""


def run_agent(user_message: str, history: list, api_key: str) -> tuple[str, list]:
    client = anthropic.Anthropic(api_key=api_key)
    messages = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages
        )

        # Ajouter la réponse à l'historique
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = " ".join(b.text for b in response.content if hasattr(b, "text"))
            return text, messages

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = TOOL_MAP.get(block.name)
                    result = fn(**block.input) if fn else f"Outil {block.name} inconnu."
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

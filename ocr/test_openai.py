import os
import time
import json
import re
import pandas as pd
from openai import OpenAI

# ===========================
# CONFIG
# ===========================
client = OpenAI()
# Dossiers
OUTPUT_JSON_DIR = "../data/output/matches_json"
OUTPUT_CSV_DIR = "../data/output/matches_csv"
GLOBAL_CSV = "../data/output/all_matches.csv"

# Modèle
MODEL = "gpt-4o"

# Prompt GPT
PROMPT = """
Tu es un OCR strict. Extrait les infos d’un tableau de stats NBA 2K.
Retourne un JSON structuré EXACTEMENT sous ce format :

{
  "match_id": "string",
  "teams": [
    {
      "team_name": "string",
      "score": int,
      "players": [
        {
          "name": "string",
          "note": "string",
          "pts": int,
          "reb": int,
          "ast": int,
          "stl": int,
          "blk": int,
          "to": int,
          "fg": "x/y",
          "3pt": "x/y",
          "ft": "x/y"
        }
      ]
    }
  ]
}

Règles strictes :
- Donne UNIQUEMENT les chiffres visibles.
- Si un chiffre est illisible ou douteux → mets null.
- Les scores d’équipe doivent être des entiers >= 0.
- Chaque joueur doit avoir une ligne, même si certaines stats = null.
"""

# ===========================
# FONCTIONS
# ===========================


def extract_match(img_url: str):
    """Envoie l'image via URL à GPT pour extraire le match"""
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
            max_tokens=3000,
        )
        elapsed = time.time() - start
        raw_json = resp.choices[0].message.content.strip()
        return raw_json, elapsed
    except Exception as e:
        return {"error": str(e)}, 0


def clean_gpt_raw_json(raw_text: str) -> dict:
    """Nettoie la sortie GPT pour obtenir un JSON utilisable"""
    if not raw_text:
        return {}

    cleaned = raw_text.strip()
    # Supprimer ```json ... ```
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    # Convertir \n et \t en vrais sauts
    cleaned = cleaned.replace("\\n", "\n").replace("\\t", "\t")
    # Retirer éventuelles guillemets autour
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].replace('\\"', '"')
    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON invalide : {e}")
        return {}


def json_to_dataframe(data: dict) -> pd.DataFrame:
    """Transforme un dict JSON en DataFrame"""
    rows = []
    for team in data.get("teams", []):
        team_name = team.get("team_name", "")
        team_score = team.get("score", 0)
        for player in team.get("players", []):
            row = {
                "match_id": data.get("match_id", ""),
                "team": team_name,
                "team_score": team_score,
                **player,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def save_json(data: dict, match_id: str) -> str:
    """Sauvegarde le JSON dans un fichier"""
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_JSON_DIR, f"{match_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def save_match_csv(df: pd.DataFrame, match_id: str) -> str:
    """Sauvegarde la DataFrame en CSV"""
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_CSV_DIR, f"{match_id}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def validate_match(df: pd.DataFrame):
    """Vérifie la cohérence du score des équipes"""
    warnings = []
    for team in df["team"].unique():
        team_rows = df[df["team"] == team]
        team_score = team_rows["team_score"].iloc[0]
        players_pts = pd.to_numeric(team_rows["pts"], errors="coerce").fillna(0).sum()
        if abs(players_pts - team_score) > 30:
            warnings.append(
                f"[WARN] {team}: score équipe ({team_score}) != somme joueurs ({players_pts})"
            )
    return warnings


# ===========================
# MAIN
# ===========================


def main():
    # Mets ici tes liens Imgur
    imgur_links = [
        "https://i.imgur.com/q42hiJB.jpeg",
        "https://i.imgur.com/EM75QHQ.jpeg",
        "https://i.imgur.com/IWYjhgi.jpeg",
    ]

    all_dfs = []

    for idx, url in enumerate(imgur_links, start=1):
        print(f"\n=== Processing match {idx} ===")
        raw, elapsed = extract_match(url)
        print(f"[INFO] GPT processing time: {elapsed:.2f}s")

        data = clean_gpt_raw_json(raw)
        if not data or "teams" not in data:
            print(f"[WARN] JSON invalide pour match {idx}")
            continue

        match_id = data.get("match_id", f"match_{idx}")

        # Save JSON
        json_path = save_json(data, match_id)
        print(f"[OK] JSON sauvegardé : {json_path}")

        # DataFrame & CSV
        df = json_to_dataframe(data)
        save_path = save_match_csv(df, match_id)
        print(f"[OK] CSV sauvegardé : {save_path}")

        # Validation
        warnings = validate_match(df)
        for w in warnings:
            print(w)

        all_dfs.append(df)

    # CSV global
    if all_dfs:
        global_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(GLOBAL_CSV), exist_ok=True)
        global_df.to_csv(GLOBAL_CSV, index=False)
        print(
            f"\n=== CSV global sauvegardé : {GLOBAL_CSV} ({len(global_df)} lignes) ==="
        )


if __name__ == "__main__":
    main()

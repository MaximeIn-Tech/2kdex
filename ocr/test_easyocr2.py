import cv2
from PIL import Image
import easyocr
import pytesseract
import pandas as pd
import numpy as np
import re

# Initialisation OCR
reader = easyocr.Reader(["en"])  # EasyOCR pour joueurs/stats

# ---------------- OCR HELPERS ----------------


def extract_text_tesseract(img, psm=7):
    """OCR avec Tesseract, spécial pour noms d'équipes (texte en majuscules propres)"""
    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(img, config=config)
    text = text.strip().replace("\n", " ")
    text = re.sub(r"[^A-Z0-9\s._-]", "", text)  # garder majuscules, chiffres
    return text


def extract_text_easyocr(img, content_type="general"):
    """OCR avec EasyOCR (joueurs/stats)"""
    try:
        results = reader.readtext(img, detail=0, paragraph=False)
        if not results:
            return ""

        if content_type == "names":
            text = " ".join(results)
            text = re.sub(r"[^a-zA-Z0-9\s._-]", "", text)

        elif content_type == "stats":
            text = " ".join(results)
            text = re.sub(r"[^0-9/:\s]", "", text)

        else:
            text = " ".join(results)

        return " ".join(text.split())

    except Exception as e:
        print(f"Erreur EasyOCR: {e}")
        return ""


def fix_fraction_easyocr(text):
    """Corrige les fractions type 2/5 ou 10/12"""
    if not text:
        return "0/0"
    text = text.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    text = text.replace(":", "/")

    matches = re.findall(r"(\d{1,2})/(\d{1,2})", text)
    if matches:
        return f"{matches[0][0]}/{matches[0][1]}"

    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2:
        return f"{numbers[0]}/{numbers[1]}"
    return "0/0"


# ---------------- EXTRACTION ----------------


def extract_row_simple(row_img, row_idx, team_name, first_line=False):
    """Extraction d'une ligne de stats"""
    h, w, _ = row_img.shape
    if first_line:
        name_zone = row_img[:, : int(w * 0.20)]
    else:
        logo_margin = int(w * 0.053)
        name_zone = row_img[:, logo_margin : int(w * 0.20)]
    stats_zone = row_img[:, int(w * 0.24) :]

    # OCR équipe (Tesseract) si première ligne
    if first_line:
        team_name_detected = extract_text_tesseract(name_zone)
        return {
            "team": team_name_detected,
            "player": "",
            "note": "",
            "pts": "0",
            "reb": "0",
            "pad": "0",
            "int": "0",
            "ctr": "0",
            "fautes": "0",
            "bp": "0",
            "tr_tt": "0/0",
            "3pr_3pt": "0/0",
            "lfr_lft": "0/0",
        }

    # OCR joueur (EasyOCR)
    player_name = extract_text_easyocr(name_zone, "names")
    if len(player_name) < 2 or not any(c.isalpha() for c in player_name):
        player_name = f"Player{row_idx}"

    # OCR stats (EasyOCR)
    stats_gray = cv2.cvtColor(stats_zone, cv2.COLOR_BGR2GRAY)
    stats_processed = cv2.resize(
        stats_gray, (stats_gray.shape[1] * 3, stats_gray.shape[0] * 3)
    )
    stats_text = extract_text_easyocr(stats_processed, "stats")
    stats_text = stats_text.replace(":", "/")
    stats_parts = stats_text.split()

    tr_tt = fix_fraction_easyocr(stats_parts[-3]) if len(stats_parts) >= 3 else "0/0"
    pr_3pt = fix_fraction_easyocr(stats_parts[-2]) if len(stats_parts) >= 2 else "0/0"
    lfr_lft = fix_fraction_easyocr(stats_parts[-1]) if len(stats_parts) >= 1 else "0/0"

    return {
        "team": team_name,
        "player": player_name,
        "note": stats_parts[0] if len(stats_parts) > 0 else "",
        "pts": stats_parts[1] if len(stats_parts) > 1 else "0",
        "reb": stats_parts[2] if len(stats_parts) > 2 else "0",
        "pad": stats_parts[3] if len(stats_parts) > 3 else "0",
        "int": stats_parts[4] if len(stats_parts) > 4 else "0",
        "ctr": stats_parts[5] if len(stats_parts) > 5 else "0",
        "fautes": stats_parts[6] if len(stats_parts) > 6 else "0",
        "bp": stats_parts[7] if len(stats_parts) > 7 else "0",
        "tr_tt": tr_tt,
        "3pr_3pt": pr_3pt,
        "lfr_lft": lfr_lft,
    }


def extract_team_data_simple(zone, team_label):
    """Extraction d'une zone complète"""
    h = zone.shape[0]
    row_height = h // 6  # suppose 6 lignes par équipe
    players_data = []

    print(f"\nLignes détectées pour {team_label}: 6")

    for idx in range(6):
        y1 = idx * row_height
        y2 = (idx + 1) * row_height
        row_img = zone[y1:y2, :]
        if idx == 0:
            # Nom de l'équipe
            team_entry = extract_row_simple(row_img, idx, team_label, first_line=True)
            team_name_detected = team_entry["team"]
        else:
            player_data = extract_row_simple(row_img, idx, team_name_detected)
            if player_data:
                players_data.append(player_data)

    return players_data


def main():
    img = cv2.imread("../data/images/image.png")
    if img is None:
        print("Erreur: impossible de charger l'image")
        return

    h, w, _ = img.shape
    print(f"Dimensions de l'image: {w}x{h}")

    # Zones des deux équipes
    team1_coords = (int(h * 0.19), int(h * 0.44), int(w * 0.30), int(w * 0.92))
    team2_coords = (int(h * 0.475), int(h * 0.73), int(w * 0.30), int(w * 0.92))

    team1_zone = img[
        team1_coords[0] : team1_coords[1], team1_coords[2] : team1_coords[3]
    ]
    team2_zone = img[
        team2_coords[0] : team2_coords[1], team2_coords[2] : team2_coords[3]
    ]

    print("\n=== Extraction hybride (Tesseract + EasyOCR) ===")

    team1_data = extract_team_data_simple(team1_zone, "Team 1")
    team2_data = extract_team_data_simple(team2_zone, "Team 2")

    all_data = team1_data + team2_data
    if not all_data:
        print("Aucune donnée extraite.")
        return None

    df = pd.DataFrame(all_data)
    print(f"\n=== Résultats finaux ({len(all_data)} joueurs) ===")
    print(df)

    df.to_csv("nba2k_stats_hybride.csv", index=False)
    print("\nDonnées sauvegardées dans 'nba2k_stats_hybride.csv'")
    return df


if __name__ == "__main__":
    main()

import cv2
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import re
import os
import easyocr
import time
from collections import Counter

# Initialisation EasyOCR (en anglais car les stats sont en chiffres)
reader = easyocr.Reader(["en"], gpu=False)


def fix_fraction(text):
    """Corrige les fractions du type 2/5"""
    if not text:
        return "0/0"
    text = (
        text.replace("O", "0")
        .replace("o", "0")
        .replace("l", "1")
        .replace("I", "1")
        .replace(":", "/")
    )
    matches = re.findall(r"\d{1,2}/\d{1,2}", text)
    if matches:
        return matches[0]
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2:
        return f"{numbers[0]}/{numbers[1]}"
    return "0/0"


def fix_zeros_ones_majority(val_list):
    """
    Prend une liste de valeurs OCRisées (ex: après 5 ou 10 passes)
    et applique un vote majoritaire pour corriger les 0/1.
    """
    if not val_list:
        return "0"

    cleaned = []
    for val in val_list:
        if not val:
            continue
        v = val.replace("O", "0").replace("o", "0")
        v = v.replace("I", "1").replace("l", "1").replace("|", "1")
        v = re.sub(r"[^0-9/]", "", v)
        if v:
            cleaned.append(v)

    if not cleaned:
        return "0"

    # Vote majoritaire
    counter = Counter(cleaned)
    most_common_val, _ = counter.most_common(1)[0]
    return most_common_val


def preprocess_for_line_detection(zone):
    """
    Transforme la zone pour détecter toutes les lignes même sur fond coloré
    - Convertit en gris
    - Fait un seuillage adaptatif pour transformer fond clair (jaune ou autre) en blanc
    - Texte noir reste noir
    """
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

    # Seuillage adaptatif (fond clair -> blanc, texte -> noir)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
    )
    return processed


def is_yellow_line(row_img, yellow_rgb=(238, 231, 0), tol=30):
    """
    Vérifie si une ligne contient un fond jaune
    - yellow_rgb = tuple RGB du jaune
    - tol = tolérance pour comparer chaque canal
    """
    if row_img.shape[2] == 3:
        b, g, r = cv2.split(row_img)
        mask = (
            (np.abs(r.astype(int) - yellow_rgb[0]) < tol)
            & (np.abs(g.astype(int) - yellow_rgb[1]) < tol)
            & (np.abs(b.astype(int) - yellow_rgb[2]) < tol)
        )
        yellow_ratio = np.sum(mask) / (row_img.shape[0] * row_img.shape[1])
        return yellow_ratio > 0.1  # si >10% pixels jaunes → ligne jaune
    return False


def preprocess_gray_text(row_img):
    """
    AMÉLIORÉ: Texte gris clair sur fond très clair/bruité → texte noir sur fond blanc
    - Débruitage agressif pour éliminer les patterns de fond
    - Lissage multi-étapes pour uniformiser le fond
    - Seuillage robuste avec double passage
    """
    # Débruitage initial fort pour éliminer patterns de transparence
    denoised = cv2.fastNlMeansDenoising(row_img, None, 10, 7, 21)

    # Lissage en 2 étapes : bilateral pour préserver contours + gaussien pour fond
    img_smooth1 = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)
    img_smooth2 = cv2.GaussianBlur(img_smooth1, (5, 5), 0)

    # Conversion en gris
    gray = (
        cv2.cvtColor(img_smooth2, cv2.COLOR_BGR2GRAY)
        if len(row_img.shape) == 3
        else img_smooth2
    )

    # Double seuillage pour éliminer le bruit résiduel
    # 1er passage : seuillage adaptatif permissif
    thresh1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 8
    )

    # 2ème passage : morphologie pour nettoyer + seuillage Otsu sur le résultat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

    # Redimensionnement pour OCR (plus agressif pour texte difficile)
    h, w = cleaned.shape
    resized = cv2.resize(cleaned, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    return resized


def preprocess_names_for_ocr(row_img):
    """
    AMÉLIORÉ: Prétraitement robuste pour OCR des noms sur fonds bruités
    - Débruitage préalable pour éliminer patterns de transparence
    - Amélioration contraste adaptative
    - Morphologie pour renforcer les caractères fins
    """
    # Débruitage initial pour éliminer le bruit de fond
    if len(row_img.shape) == 3:
        denoised = cv2.fastNlMeansDenoising(row_img, None, 8, 7, 21)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.fastNlMeansDenoising(row_img, None, 8, 7, 21)

    # Lissage adaptatif pour uniformiser le fond sans perdre le texte
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Amélioration du contraste avec CLAHE local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Étirement du contraste plus agressif pour séparer texte/fond
    min_val, max_val = np.percentile(enhanced, 2), np.percentile(enhanced, 98)
    contrast = np.clip((enhanced - min_val) * 255 / (max_val - min_val), 0, 255).astype(
        np.uint8
    )

    # Morphologie pour renforcer les traits fins (underscores, etc.)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1))
    contrast = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, kernel)

    # Upscale (plus important pour texte sur fond bruité)
    h, w = contrast.shape
    resized = cv2.resize(contrast, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    return resized


def extract_scoreboard(img):
    """
    Extrait les scores avec EasyOCR (haut=team1, bas=team2).
    """
    h, w, _ = img.shape
    scoreboard_zone = img[int(h * 0.33) : int(h * 0.61), int(w * 0.19) : int(w * 0.25)]
    sh, sw, _ = scoreboard_zone.shape

    team1_zone = scoreboard_zone[0 : int(sh * 0.47), :]
    team2_zone = scoreboard_zone[int(sh * 0.5) :, :]

    def ocr_score(zone, filename):
        cv2.imwrite(filename, zone)
        results = reader.readtext(zone, detail=0, allowlist="0123456789")
        return results[0] if results else "0"

    team1_score = ocr_score(team1_zone, "debug_team1_easy.png")
    team2_score = ocr_score(team2_zone, "debug_team2_easy.png")

    return {"team1_score": team1_score, "team2_score": team2_score}


def simple_resize_only(img, scale=3):
    """
    AMÉLIORÉ: Redimensionnement avec débruitage préalable
    """
    # Débruitage avant redimensionnement pour éviter d'amplifier le bruit
    if len(img.shape) == 3 and img.shape[2] == 3:
        denoised = cv2.fastNlMeansDenoising(img, None, 8, 7, 21)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    else:
        gray = (
            cv2.fastNlMeansDenoising(img, None, 8, 7, 21)
            if len(img.shape) == 3
            else img
        )

    height, width = gray.shape
    resized = cv2.resize(
        gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC
    )
    return resized


# def extract_row_simple(
#     row_img, row_idx, team_name, first_line=False, output_dir="debug_cols"
# ):
#     """Extraction OCR hybride avec sauvegarde des zones nom et stats"""

#     import os

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     h, w, _ = row_img.shape

#     # --- Zone nom ---
#     if first_line:
#         name_zone = row_img[:, : int(w * 0.20)]
#     else:
#         logo_margin = int(w * 0.053)
#         name_zone = row_img[:, logo_margin : int(w * 0.20)]

#     # --- Zone stats ---
#     stats_zone = row_img[:, int(w * 0.24) :]

#     # --- Sauvegarde des zones complètes ---
#     cv2.imwrite(f"{output_dir}/{team_name}_row{row_idx}_full.png", row_img)
#     cv2.imwrite(f"{output_dir}/{team_name}_row{row_idx}_name_zone.png", name_zone)
#     cv2.imwrite(f"{output_dir}/{team_name}_row{row_idx}_stats_zone.png", stats_zone)

#     # --- OCR noms avec Tesseract ---
#     name_processed = cv2.cvtColor(name_zone, cv2.COLOR_BGR2GRAY)
#     name_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./+-_: "
#     player_name = pytesseract.image_to_string(
#         name_processed, config=name_config
#     ).strip()
#     player_name = " ".join(player_name.split())

#     if first_line:
#         return {
#             "team": player_name,
#             "player": "",
#             "note": "",
#             "pts": "",
#             "reb": "",
#             "pad": "",
#             "int": "",
#             "ctr": "",
#             "fautes": "",
#             "bp": "",
#             "tr_tt": "",
#             "3pr_3pt": "",
#             "lfr_lft": "",
#         }

#     # --- Découpage des colonnes de stats ---
#     col_ratios = [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.66, 0.75, 0.90, 1.0]
#     stats_values = []

#     for i in range(len(col_ratios) - 1):
#         x1 = int(stats_zone.shape[1] * col_ratios[i])
#         x2 = int(stats_zone.shape[1] * col_ratios[i + 1])
#         col_img = stats_zone[:, x1:x2]
#         col_gray = (
#             cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
#             if len(col_img.shape) == 3
#             else col_img
#         )
#         col_gray = cv2.resize(
#             col_gray,
#             (col_gray.shape[1] * 4, col_gray.shape[0] * 4),
#             interpolation=cv2.INTER_CUBIC,
#         )

#         # --- Sauvegarde de chaque colonne ---
#         cv2.imwrite(f"{output_dir}/{team_name}_row{row_idx}_col{i}.png", col_gray)

#         val_list = reader.readtext(col_gray, detail=0, paragraph=False)
#         val = val_list[0].strip() if val_list else "0"

#         # Colonne note = A-G + +/-
#         if i == 0:
#             val = val.upper()
#             val = re.sub(r"[^A-G+-_]", "", val)
#         # Autres stats = numériques et fractions seulement
#         else:
#             val = re.sub(r"[^0-9/]", "", val)
#             val = (
#                 val.replace("O", "0")
#                 .replace("o", "0")
#                 .replace("l", "1")
#                 .replace("I", "1")
#                 .replace(":", "/")
#             )

#         stats_values.append(val)

#     # Validation nom joueur
#     if len(player_name) < 2 or not any(c.isalpha() for c in player_name):
#         player_name = f"Player{row_idx}"

#     # Correction fractions
#     tr_tt = fix_fraction(stats_values[-3]) if len(stats_values) >= 3 else "0/0"
#     pr_3pt = fix_fraction(stats_values[-2]) if len(stats_values) >= 2 else "0/0"
#     lfr_lft = fix_fraction(stats_values[-1]) if len(stats_values) >= 1 else "0/0"

#     return {
#         "team": team_name,
#         "player": player_name,
#         "note": stats_values[0] if len(stats_values) > 0 else "",
#         "pts": stats_values[1] if len(stats_values) > 1 else "0",
#         "reb": stats_values[2] if len(stats_values) > 2 else "0",
#         "pad": stats_values[3] if len(stats_values) > 3 else "0",
#         "int": stats_values[4] if len(stats_values) > 4 else "0",
#         "ctr": stats_values[5] if len(stats_values) > 5 else "0",
#         "fautes": stats_values[6] if len(stats_values) > 6 else "0",
#         "bp": stats_values[7] if len(stats_values) > 7 else "0",
#         "tr_tt": tr_tt,
#         "3pr_3pt": pr_3pt,
#         "lfr_lft": lfr_lft,
#     }


# def extract_row_easyocr(
#     row_img, row_idx, team_name, first_line=False, output_dir="debug_cols"
# ):
#     """
#     Extraction OCR avec EasyOCR : noms et stats.
#     """
#     import os

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     h, w, _ = row_img.shape

#     # --- Zone nom ---
#     if first_line:
#         name_zone = row_img[:, : int(w * 0.20)]
#     else:
#         logo_margin = int(w * 0.053)
#         name_zone = row_img[:, logo_margin : int(w * 0.20)]

#     # --- Zone stats ---
#     stats_zone = row_img[:, int(w * 0.24) :]

#     # --- OCR noms avec EasyOCR ---
#     name_processed = preprocess_names_for_ocr(name_zone)
#     name_results = reader.readtext(name_processed, detail=0)
#     player_name = name_results[0].strip() if name_results else f"Player{row_idx}"
#     player_name = " ".join(player_name.split())

#     if first_line:
#         return {
#             "team": player_name,
#             "player": "",
#             "note": "",
#             "pts": "",
#             "reb": "",
#             "pad": "",
#             "int": "",
#             "ctr": "",
#             "fautes": "",
#             "bp": "",
#             "tr_tt": "",
#             "3pr_3pt": "",
#             "lfr_lft": "",
#         }

#     # --- Découpage des colonnes de stats ---
#     col_ratios = [
#         0,
#         0.08,
#         0.16,
#         0.24,
#         0.32,
#         0.40,
#         0.48,
#         0.56,
#         0.66,
#         0.75,
#         0.90,
#         1.0,
#     ]
#     stats_values = []

#     for i in range(len(col_ratios) - 1):
#         x1 = int(stats_zone.shape[1] * col_ratios[i])
#         x2 = int(stats_zone.shape[1] * col_ratios[i + 1])
#         col_img = stats_zone[:, x1:x2]
#         col_gray = (
#             cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
#             if len(col_img.shape) == 3
#             else col_img
#         )
#         col_gray = cv2.resize(
#             col_gray,
#             (col_gray.shape[1] * 4, col_gray.shape[0] * 4),
#             interpolation=cv2.INTER_CUBIC,
#         )
#         cv2.imwrite(f"{output_dir}/{team_name}_row{row_idx}_col{i}.png", col_gray)

#         val_list = reader.readtext(col_gray, detail=0)
#         val = val_list[0].strip() if val_list else "0"

#         # Colonne note = A-G + +/-
#         if i == 0:
#             val = val.upper()
#             val = re.sub(r"[^A-G+-_]", "", val)
#         else:
#             val = re.sub(r"[^0-9/]", "", val)
#             val = (
#                 val.replace("O", "0")
#                 .replace("o", "0")
#                 .replace("l", "1")
#                 .replace("I", "1")
#                 .replace(":", "/")
#             )

#         stats_values.append(val)

#     # Correction fractions
#     tr_tt = fix_fraction(stats_values[-3]) if len(stats_values) >= 3 else "0/0"
#     pr_3pt = fix_fraction(stats_values[-2]) if len(stats_values) >= 2 else "0/0"
#     lfr_lft = fix_fraction(stats_values[-1]) if len(stats_values) >= 1 else "0/0"

#     return {
#         "team": team_name,
#         "player": player_name,
#         "note": stats_values[0] if len(stats_values) > 0 else "",
#         "pts": stats_values[1] if len(stats_values) > 1 else "0",
#         "reb": stats_values[2] if len(stats_values) > 2 else "0",
#         "pad": stats_values[3] if len(stats_values) > 3 else "0",
#         "int": stats_values[4] if len(stats_values) > 4 else "0",
#         "ctr": stats_values[5] if len(stats_values) > 5 else "0",
#         "fautes": stats_values[6] if len(stats_values) > 6 else "0",
#         "bp": stats_values[7] if len(stats_values) > 7 else "0",
#         "tr_tt": tr_tt,
#         "3pr_3pt": pr_3pt,
#         "lfr_lft": lfr_lft,
#     }


def extract_row_simple_postprocess(
    row_img, row_idx, team_name, first_line=False, output_dir="debug_cols", n_passes=5
):
    """Extraction OCR hybride avec Tesseract et post-processing multi-pass uniquement pour les 0/1"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    h, w, _ = row_img.shape

    # --- Zone nom ---
    if first_line:
        name_zone = row_img[:, : int(w * 0.20)]
    else:
        logo_margin = int(w * 0.053)
        name_zone = row_img[:, logo_margin : int(w * 0.20)]

    # --- Zone stats ---
    stats_zone = row_img[:, int(w * 0.24) :]

    # --- OCR noms ---
    name_processed = cv2.cvtColor(name_zone, cv2.COLOR_BGR2GRAY)
    name_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./+-_: "
    player_name = pytesseract.image_to_string(
        name_processed, config=name_config
    ).strip()
    player_name = " ".join(player_name.split())

    if first_line:
        return {
            "team": player_name,
            "player": "",
            "note": "",
            "pts": "",
            "reb": "",
            "pad": "",
            "int": "",
            "ctr": "",
            "fautes": "",
            "bp": "",
            "tr_tt": "",
            "3pr_3pt": "",
            "lfr_lft": "",
        }

    # --- Découpage des colonnes de stats ---
    col_ratios = [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.66, 0.75, 0.90, 1.0]
    stats_values = []

    for i in range(len(col_ratios) - 1):
        x1 = int(stats_zone.shape[1] * col_ratios[i])
        x2 = int(stats_zone.shape[1] * col_ratios[i + 1])
        col_img = stats_zone[:, x1:x2]
        col_gray = (
            cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
            if len(col_img.shape) == 3
            else col_img
        )
        col_gray = cv2.resize(
            col_gray,
            (col_gray.shape[1] * 4, col_gray.shape[0] * 4),
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imwrite(f"{output_dir}/{team_name}_row{row_idx}_col{i}.png", col_gray)

        # --- OCR multi-pass uniquement si c'est un score binaire (0/1) ---
        val_candidates = []
        # Lecture initiale
        val = pytesseract.image_to_string(col_gray, config=r"--oem 3 --psm 7").strip()

        if i == 0:
            # Colonne note = A-G + +/-
            val = val.upper()
            val = re.sub(r"[^A-G+-_]", "", val)
        else:
            # Stats numériques/fractions
            val = re.sub(r"[^0-9/]", "", val)
            val = (
                val.replace("O", "0")
                .replace("o", "0")
                .replace("l", "1")
                .replace("I", "1")
                .replace(":", "/")
            )

        # Si c'est 0 ou 1, refaire n_passes et faire un vote majoritaire
        if val in ["0", "1"]:
            val_candidates.append(val)
            for _ in range(n_passes - 1):
                tmp_val = pytesseract.image_to_string(
                    col_gray, config=r"--oem 3 --psm 7"
                ).strip()
                tmp_val = re.sub(r"[^01]", "", tmp_val) or "0"
                val_candidates.append(tmp_val)
            val = Counter(val_candidates).most_common(1)[0][0]

        stats_values.append(val)

    # Validation nom joueur
    if len(player_name) < 2 or not any(c.isalpha() for c in player_name):
        player_name = f"Player{row_idx}"

    # Correction fractions
    tr_tt = fix_fraction(stats_values[-3]) if len(stats_values) >= 3 else "0/0"
    pr_3pt = fix_fraction(stats_values[-2]) if len(stats_values) >= 2 else "0/0"
    lfr_lft = fix_fraction(stats_values[-1]) if len(stats_values) >= 1 else "0/0"

    return {
        "team": team_name,
        "player": player_name,
        "note": stats_values[0] if len(stats_values) > 0 else "",
        "pts": stats_values[1] if len(stats_values) > 1 else "0",
        "reb": stats_values[2] if len(stats_values) > 2 else "0",
        "pad": stats_values[3] if len(stats_values) > 3 else "0",
        "int": stats_values[4] if len(stats_values) > 4 else "0",
        "ctr": stats_values[5] if len(stats_values) > 5 else "0",
        "fautes": stats_values[6] if len(stats_values) > 6 else "0",
        "bp": stats_values[7] if len(stats_values) > 7 else "0",
        "tr_tt": tr_tt,
        "3pr_3pt": pr_3pt,
        "lfr_lft": lfr_lft,
    }


def detect_text_rows_simple(zone):
    """Détection des lignes après preprocessing pour neutraliser le jaune"""

    processed = preprocess_for_line_detection(zone)

    # Calcul de la variance par ligne (plus robuste après seuillage)
    row_variances = np.var(processed, axis=1)

    # Lissage
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(row_variances, kernel, mode="same")

    # Seuil permissif
    threshold = np.mean(smoothed) * 0.5
    text_rows = smoothed > threshold

    # Groupement des lignes
    row_groups = []
    start = None
    min_height = 5  # plus petit, après seuillage on détecte mieux

    for i, is_text in enumerate(text_rows):
        if is_text and start is None:
            start = i
        elif not is_text and start is not None:
            if i - start > min_height:
                row_groups.append((start, i))
            start = None

    if start is not None:
        row_groups.append((start, len(text_rows)))

    return row_groups


def extract_team_data_simple(zone, team_name):
    """Extraction avec première ligne = nom d'équipe et toutes les autres lignes comme joueurs"""

    row_groups = detect_text_rows_simple(zone)
    print(f"\nLignes détectées pour {team_name}: {len(row_groups)}")

    if not row_groups:
        return []

    players_data = []

    # Extraire le nom de l'équipe depuis la première ligne
    first_y1, first_y2 = row_groups[0]
    first_line_img = zone[first_y1:first_y2, :]
    team_name_detected = extract_row_simple_postprocess(
        first_line_img, 0, team_name, first_line=True
    )["team"]

    # Itérer sur toutes les lignes suivantes pour les joueurs
    for idx, (y1, y2) in enumerate(row_groups[1:], start=1):
        margin = 3
        row_img = zone[max(0, y1 - margin) : min(zone.shape[0], y2 + margin), :]
        if row_img.shape[0] < 10:
            continue

        player_data = extract_row_simple_postprocess(row_img, idx, team_name_detected)
        if player_data:
            players_data.append(player_data)

    return players_data


def test_direct_ocr():
    """Test OCR direct sur l'image complète pour voir ce qui est détecté"""
    img = cv2.imread("../data/images/image5.jpeg")
    if img is None:
        print("Image non trouvée")
        return

    # OCR sur l'image complète
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=r"--oem 3 --psm 6")

    print("=== OCR COMPLET DE L'IMAGE ===")
    print(text)
    print("=" * 50)


def main():
    start_time = time.time()  # début du chronomètre

    # Chargement de l'image
    img = cv2.imread("../data/images/image5.jpeg")
    if img is None:
        print("Erreur: impossible de charger l'image")
        return

    h, w, _ = img.shape
    print(f"Dimensions originales de l'image: {w}x{h}")

    # # --- Resize pour le scoreboard seulement ---
    # target_w, target_h = 2048, 1152
    # img_for_scoreboard = img.copy()
    # if (w, h) != (target_w, target_h):
    #     img_for_scoreboard = cv2.resize(
    #         img_for_scoreboard, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    #     )
    #     print(f"Image pour scoreboard redimensionnée à: {target_w}x{target_h}")

    # Coordonnées des tableaux (utiliser l'image originale pour stats)
    team1_coords = (int(h * 0.19), int(h * 0.44), int(w * 0.30), int(w * 0.92))
    team2_coords = (int(h * 0.475), int(h * 0.73), int(w * 0.30), int(w * 0.92))

    # Extraction des zones pour stats
    team1_zone = img[
        team1_coords[0] : team1_coords[1], team1_coords[2] : team1_coords[3]
    ]
    team2_zone = img[
        team2_coords[0] : team2_coords[1], team2_coords[2] : team2_coords[3]
    ]

    # Extraction simple
    team1_data = extract_team_data_simple(team1_zone, "Team 1")
    team2_data = extract_team_data_simple(team2_zone, "Team 2")

    if not team1_data or not team2_data:
        print("Aucune donnée extraite.")
        return None

    # Noms détectés
    team1_name = team1_data[0]["team"]
    team2_name = team2_data[0]["team"]

    all_data = team1_data + team2_data

    # Extraction scoreboard sur l'image redimensionnée
    scores = extract_scoreboard(img)
    print("\n=== SCORES ===")
    scores_with_names = {
        team1_name: scores["team1_score"],
        team2_name: scores["team2_score"],
    }
    print(scores_with_names)

    # Création du DataFrame
    df = pd.DataFrame(all_data)
    print(f"\n=== Résultats finaux ({len(all_data)} joueurs) ===")
    print(df)

    # Ajout du score au CSV
    df["team1_name"] = team1_name
    df["team1_score"] = scores["team1_score"]
    df["team2_name"] = team2_name
    df["team2_score"] = scores["team2_score"]
    df.to_csv("nba2k_stats.csv", index=False)
    print("\nDonnées sauvegardées dans 'nba2k_stats.csv'")

    end_time = time.time()  # fin du chronomètre
    elapsed = end_time - start_time
    print(f"\nTemps d'exécution total: {elapsed:.2f} secondes")

    return df


if __name__ == "__main__":
    # Test OCR complet d'abord pour voir ce qui est détectable
    test_direct_ocr()

    # Puis extraction structurée
    df = main()

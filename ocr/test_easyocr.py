import cv2
from PIL import Image
import easyocr
import pandas as pd
import numpy as np
import re

# Initialisation EasyOCR (une seule fois au début)
reader = easyocr.Reader(["en"])  # Tu peux ajouter 'fr' si besoin


def fix_fraction(text):
    # Détecte uniquement les fractions 0-99 / 0-99
    matches = re.findall(r"\b([0-9]|[1-9][0-9]?)/([0-9]|[1-9][0-9]?)\b", text)
    if matches:
        # Retourne la première fraction trouvée, format "num/den"
        return f"{matches[0][0]}/{matches[0][1]}"
    return text


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
    Texte gris clair sur fond très clair → texte noir sur fond blanc
    Pour EasyOCR, on peut être moins agressif car il gère mieux le bruit
    """
    # Lissage léger
    img_smooth = cv2.bilateralFilter(row_img, d=5, sigmaColor=50, sigmaSpace=50)

    # Conversion en gris
    gray = (
        cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
        if len(row_img.shape) == 3
        else row_img
    )

    # Seuillage adaptatif moins agressif pour EasyOCR
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # EasyOCR préfère des images moins agrandies
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    return gray


def preprocess_names_for_ocr(row_img):
    """
    Prétraitement léger pour EasyOCR des noms de joueurs
    EasyOCR est plus robuste, on peut être moins agressif
    """
    # Conversion en gris si nécessaire
    gray = (
        cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
        if len(row_img.shape) == 3
        else row_img
    )

    # Débruitage léger
    denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)

    # Amélioration contraste modérée
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Upscale léger (EasyOCR préfère moins d'agrandissement)
    h, w = enhanced.shape
    resized = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    return resized


def simple_resize_only(img, scale=2):  # Scale réduit pour EasyOCR
    """Redimensionnement simple, accepte image déjà en gris"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    height, width = gray.shape
    resized = cv2.resize(
        gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC
    )
    return resized


def extract_text_easyocr(img, content_type="general"):
    """
    Extraction OCR avec EasyOCR
    - img: image préprocessée
    - content_type: "names" ou "stats" pour différentes stratégies
    """
    try:
        # EasyOCR sur l'image
        results = reader.readtext(img, detail=0, paragraph=False)

        if not results:
            return ""

        if content_type == "names":
            # Pour les noms : prendre tout le texte détecté
            text = " ".join(results)
            # Nettoyage basique
            text = re.sub(r"[^a-zA-Z0-9\s._-]", "", text)

        elif content_type == "stats":
            # Pour les stats : concaténer tous les éléments détectés
            text = " ".join(results)
            # Garder seulement chiffres, /, :, et espaces
            text = re.sub(r"[^0-9/:\s]", "", text)

        else:
            text = " ".join(results)

        # Nettoyage des espaces multiples
        text = " ".join(text.split())
        return text

    except Exception as e:
        print(f"Erreur EasyOCR: {e}")
        return ""


def extract_row_simple(row_img, row_idx, team_name, first_line=False):
    """Extraction optimisée avec EasyOCR"""

    h, w, _ = row_img.shape
    if first_line:
        name_zone = row_img[:, : int(w * 0.20)]
    else:
        logo_margin = int(w * 0.053)
        name_zone = row_img[:, logo_margin : int(w * 0.20)]
    stats_zone = row_img[:, int(w * 0.24) : int(w * 1)]

    # --- Prétraitement noms (plus léger pour EasyOCR) ---
    if is_yellow_line(name_zone):
        name_processed = simple_resize_only(preprocess_gray_text(name_zone), scale=2)
    else:
        name_processed = preprocess_names_for_ocr(name_zone)

    # --- Prétraitement stats (adapté pour EasyOCR) ---
    stats_gray = (
        cv2.cvtColor(stats_zone, cv2.COLOR_BGR2GRAY)
        if len(stats_zone.shape) == 3
        else stats_zone
    )

    # Débruitage léger
    stats_denoised = cv2.fastNlMeansDenoising(stats_gray, None, 5, 7, 21)

    # Amélioration contraste modérée
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    stats_processed = clahe.apply(stats_denoised)

    # Upscale moins agressif pour EasyOCR
    stats_processed = cv2.resize(
        stats_processed,
        (stats_processed.shape[1] * 3, stats_processed.shape[0] * 3),
        interpolation=cv2.INTER_CUBIC,
    )

    # Morphologie légère
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    stats_processed = cv2.morphologyEx(stats_processed, cv2.MORPH_CLOSE, kernel)

    # Debug
    cv2.imwrite(f"debug_{team_name}_row{row_idx}_names.png", name_processed)
    cv2.imwrite(f"debug_{team_name}_row{row_idx}_stats.png", stats_processed)

    try:
        # --- OCR noms avec EasyOCR ---
        player_name = extract_text_easyocr(name_processed, "names")

        if first_line:
            return {
                "team": player_name,
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

        # --- OCR stats avec EasyOCR ---
        stats_text = extract_text_easyocr(stats_processed, "stats")

        # Post-processing spécifique pour les stats
        stats_text = stats_text.replace(":", "/").replace("O", "0").replace("o", "0")
        stats_parts = stats_text.split() if stats_text else []

        # Validation du nom
        if len(player_name) < 2 or not any(c.isalpha() for c in player_name):
            player_name = f"Player{row_idx}"

        # --- Correction post-OCR pour fractions ---
        def fix_fraction_easyocr(text):
            if not text:
                return "0/0"

            # Correction des caractères mal reconnus
            text = (
                text.replace("O", "0")
                .replace("o", "0")
                .replace("l", "1")
                .replace("I", "1")
            )
            text = text.replace(":", "/")

            # Recherche de fraction
            matches = re.findall(r"(\d{1,2})/(\d{1,2})", text)
            if matches:
                return f"{matches[0][0]}/{matches[0][1]}"

            # Fallback: détecter chiffres séparés par espaces
            numbers = re.findall(r"\d+", text)
            if len(numbers) >= 2:
                return f"{numbers[0]}/{numbers[1]}"
            return "0/0"

        # Traitement des fractions en fin de stats
        tr_tt = (
            fix_fraction_easyocr(stats_parts[-3]) if len(stats_parts) >= 3 else "0/0"
        )
        pr_3pt = (
            fix_fraction_easyocr(stats_parts[-2]) if len(stats_parts) >= 2 else "0/0"
        )
        lfr_lft = (
            fix_fraction_easyocr(stats_parts[-1]) if len(stats_parts) >= 1 else "0/0"
        )

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

    except Exception as e:
        print(f"Erreur OCR ligne {row_idx}: {e}")
        return None


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
    team_name_detected = extract_row_simple(
        first_line_img, 0, team_name, first_line=True
    )["team"]

    # Itérer sur toutes les lignes suivantes pour les joueurs
    for idx, (y1, y2) in enumerate(row_groups[1:], start=1):
        margin = 3
        row_img = zone[max(0, y1 - margin) : min(zone.shape[0], y2 + margin), :]
        if row_img.shape[0] < 10:
            continue

        player_data = extract_row_simple(row_img, idx, team_name_detected)
        if player_data:
            players_data.append(player_data)

    return players_data


def main():
    # Chargement de l'image
    img = cv2.imread("../data/images/image.png")
    if img is None:
        print("Erreur: impossible de charger l'image")
        return

    h, w, _ = img.shape
    print(f"Dimensions de l'image: {w}x{h}")

    # Coordonnées des tableaux (ajustez si nécessaire)
    team1_coords = (int(h * 0.19), int(h * 0.44), int(w * 0.30), int(w * 0.92))
    team2_coords = (int(h * 0.475), int(h * 0.73), int(w * 0.30), int(w * 0.92))

    # Extraction des zones
    team1_zone = img[
        team1_coords[0] : team1_coords[1], team1_coords[2] : team1_coords[3]
    ]
    team2_zone = img[
        team2_coords[0] : team2_coords[1], team2_coords[2] : team2_coords[3]
    ]

    # Sauvegarde des zones complètes pour debug
    cv2.imwrite("debug_team1_zone_original.png", team1_zone)
    cv2.imwrite("debug_team2_zone_original.png", team2_zone)

    print("\n=== Extraction avec EasyOCR ===")

    # Extraction simple
    team1_data = extract_team_data_simple(team1_zone, "Team 1")
    team2_data = extract_team_data_simple(team2_zone, "Team 2")

    # Compilation des résultats
    all_data = team1_data + team2_data

    if not all_data:
        print("Aucune donnée extraite.")
        return None

    # Création du DataFrame
    df = pd.DataFrame(all_data)
    print(f"\n=== Résultats finaux ({len(all_data)} joueurs) ===")
    print(df)

    # Sauvegarde
    df.to_csv("nba2k_stats_easyocr.csv", index=False)
    print("\nDonnées sauvegardées dans 'nba2k_stats_easyocr.csv'")

    return df


def test_direct_easyocr():
    """Test EasyOCR direct sur l'image complète pour voir ce qui est détecté"""
    img = cv2.imread("../data/images/image.png")
    if img is None:
        print("Image non trouvée")
        return

    print("=== EasyOCR COMPLET DE L'IMAGE ===")
    results = reader.readtext(img, detail=0)
    for text in results:
        print(text)
    print("=" * 50)


if __name__ == "__main__":
    print("Initialisation EasyOCR...")

    # Test EasyOCR complet d'abord pour voir ce qui est détectable
    test_direct_easyocr()

    # Puis extraction structurée
    df = main()

import cv2
import pytesseract
import pandas as pd
import numpy as np


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


def preprocess_for_ocr(row_img):
    """
    Prétraitement OCR pour texte gris clair sur fond clair
    - Inverse le gris si texte clair
    - Egalisation d'histogramme
    - Légère netteté
    - Seuillage adaptatif
    """
    gray = (
        cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
        if len(row_img.shape) == 3
        else row_img
    )

    if np.mean(gray) > 180:
        gray = 255 - gray

    gray = cv2.equalizeHist(gray)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    return processed


def simple_resize_only(img, scale=3):
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


def extract_row_simple(row_img, row_idx, team_name, first_line=False):
    """Extraction simple avec gestion du texte noir sur jaune et première ligne = nom d'équipe"""

    h, w, _ = row_img.shape
    name_zone = row_img[:, : int(w * 0.20)]
    stats_zone = row_img[:, int(w * 0.20) :]

    # Prétraitement : neutralisation du jaune + redimensionnement
    name_processed = simple_resize_only((name_zone), scale=4)
    stats_processed = simple_resize_only((stats_zone), scale=3)

    # Debug
    cv2.imwrite(f"debug_{team_name}_row{row_idx}_names.png", name_processed)
    cv2.imwrite(f"debug_{team_name}_row{row_idx}_stats.png", stats_processed)

    try:
        # OCR noms
        name_config = r"--oem 3 --psm 7"
        player_name = pytesseract.image_to_string(
            name_processed, config=name_config
        ).strip()
        player_name = " ".join(player_name.split())

        if first_line:
            # Première ligne = nom de l'équipe
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

        # OCR stats
        stats_config = (
            r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFabcdef./-+: "
        )
        stats_text = pytesseract.image_to_string(
            stats_processed, config=stats_config
        ).strip()
        stats_text = " ".join(stats_text.split())
        stats_parts = stats_text.split()

        # Validation du nom
        if len(player_name) < 2 or not any(c.isalpha() for c in player_name):
            player_name = f"Player{row_idx}"

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
            "tr_tt": stats_parts[8] if len(stats_parts) > 8 else "0/0",
            "3pr_3pt": stats_parts[9] if len(stats_parts) > 9 else "0/0",
            "lfr_lft": stats_parts[10] if len(stats_parts) > 10 else "0/0",
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

    print("\n=== Extraction SIMPLE (sans preprocessing) ===")

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
    df.to_csv("nba2k_stats.csv", index=False)
    print("\nDonnées sauvegardées dans 'nba2k_stats.csv'")

    return df


def test_direct_ocr():
    """Test OCR direct sur l'image complète pour voir ce qui est détecté"""
    img = cv2.imread("../data/images/image.png")
    if img is None:
        print("Image non trouvée")
        return

    # OCR sur l'image complète
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=r"--oem 3 --psm 6")

    print("=== OCR COMPLET DE L'IMAGE ===")
    print(text)
    print("=" * 50)


if __name__ == "__main__":
    # Test OCR complet d'abord pour voir ce qui est détectable
    test_direct_ocr()

    # Puis extraction structurée
    df = main()

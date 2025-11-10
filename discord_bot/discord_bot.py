import discord
import os
import aiohttp
from ocr.test2 import main
from discord.ui import View, Button

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GUILD_ID = 1295459991452651540  # ID du serveur
CHANNEL_ID = 1423652421519016049  # ID du channel à écouter
SAVE_DIR = "/Volumes/Maxime HDD/2. PROFESSIONNEL/1. Programmation/2kdex/data/images"
CSV_PATH = os.path.join(SAVE_DIR, "all_stats.csv")  # CSV cumulatif
# ---------------------------------------

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

client = discord.Client(intents=intents)

# Crée le dossier de sauvegarde si nécessaire
os.makedirs(SAVE_DIR, exist_ok=True)


@client.event
async def on_ready():
    print(f"{client.user} connecté et prêt!")


class ExportCSVView(View):
    def __init__(self, csv_path):
        super().__init__(timeout=None)
        self.csv_path = csv_path

    @discord.ui.button(label="📤 Exporter en CSV", style=discord.ButtonStyle.primary)
    async def export_csv(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        user = interaction.user
        try:
            if not os.path.exists(self.csv_path):
                await interaction.response.send_message(
                    "⚠️ Aucun fichier CSV trouvé.", ephemeral=True
                )
                return

            await interaction.response.send_message(
                "📩 Envoi du CSV en message privé...", ephemeral=True
            )

            # Envoi du fichier en DM
            await user.send(
                "Voici le CSV complet avec toutes les statistiques :",
                file=discord.File(self.csv_path),
            )
        except discord.Forbidden:
            await interaction.followup.send(
                "❌ Impossible d’envoyer un DM. Active tes messages privés pour ce serveur.",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.followup.send(
                f"❌ Erreur lors de l’envoi du CSV : {e}", ephemeral=True
            )


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.channel.id != CHANNEL_ID:
        return

    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(SAVE_DIR, attachment.filename)

                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status == 200:
                            with open(file_path, "wb") as f:
                                f.write(await resp.read())
                            await message.channel.send(
                                f"Image téléchargée : {attachment.filename}"
                            )
                            await message.channel.send(
                                "Reconnaissance des scores en cours... Merci de patienter !"
                            )

                            df = main(file_path)  # retourne le DataFrame complet
                            if df is not None and not df.empty:
                                # Récupérer le score
                                team1_name = df["team1_name"].iloc[0]
                                team2_name = df["team2_name"].iloc[0]
                                team1_score = df["team1_score"].iloc[0]
                                team2_score = df["team2_score"].iloc[0]

                                await message.channel.send(
                                    f"Score du match : **{team1_name} {team1_score} - {team2_score} {team2_name}**"
                                )

                                # Convertir le DataFrame en Markdown table pour Discord
                                # Limite à 10 premières lignes si trop long
                                df_to_show = df.drop(
                                    columns=[
                                        "team1_name",
                                        "team1_score",
                                        "team2_name",
                                        "team2_score",
                                    ]
                                ).head(10)
                                table_str = (
                                    "```\n"
                                    + df_to_show.to_string(index=False)
                                    + "\n```"
                                )

                                # Sauvegarde cumulatif CSV
                                df.to_csv(
                                    CSV_PATH,
                                    mode="a",
                                    index=False,
                                    header=not os.path.exists(CSV_PATH),
                                )

                                view = ExportCSVView(CSV_PATH)
                                await message.channel.send(
                                    f"Score du match : **{team1_name} {team1_score} - {team2_score} {team2_name}**",
                                    view=view,
                                )
                            else:
                                await message.channel.send(
                                    f"Aucune donnée extraite pour {attachment.filename}"
                                )


client.run(TOKEN)

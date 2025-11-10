import discord
from discord.ext import commands
import os
import aiohttp
from ocr.test2 import main
from discord.ui import View
from dotenv import load_dotenv
import io

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GUILD_ID = 1295459991452651540
CHANNEL_ID = 1423652421519016049
SAVE_DIR = os.getenv("SAVE_DIR")
CSV_PATH = os.path.join(SAVE_DIR, "all_stats.csv")

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

client = commands.Bot(command_prefix="!", intents=intents)
os.makedirs(SAVE_DIR, exist_ok=True)


@client.event
async def on_ready():
    print(f"{client.user} connecté et prêt!")


class ExportMatchCSVView(View):
    def __init__(self, df):
        super().__init__(timeout=None)
        self.df = df

    @discord.ui.button(
        label="📤 Télécharger ce match en CSV", style=discord.ButtonStyle.primary
    )
    async def export_csv(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        try:
            # Créer un fichier CSV en mémoire
            csv_buffer = io.StringIO()
            self.df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            # Créer un fichier Discord à partir du buffer
            csv_file = discord.File(
                fp=io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
                filename=f"match_stats.csv",
            )

            await interaction.response.send_message(
                "📊 Voici les statistiques de ce match :", file=csv_file, ephemeral=True
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ Erreur lors de l'export : {str(e)}",
                ephemeral=True,
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

                await message.channel.send(f"Image téléchargée : {attachment.filename}")
                await message.channel.send(
                    "Reconnaissance des scores en cours... Merci de patienter !"
                )

                df = main(file_path)

                if df is not None and not df.empty:
                    # Sauvegarder dans le CSV global
                    df.to_csv(
                        CSV_PATH,
                        mode="a",
                        index=False,
                        header=not os.path.exists(CSV_PATH),
                    )

                    # Formatter le DataFrame pour l'affichage
                    df_display = df.to_string(index=False)

                    # Envoyer le DataFrame formaté
                    await message.channel.send(f"```\n{df_display}\n```")

                    # Envoyer le bouton d'export en dessous
                    view = ExportMatchCSVView(df)
                    await message.channel.send(
                        "📥 Télécharge les stats de ce match :", view=view
                    )


# IMPORTANT : client.run() doit être HORS de la fonction on_message !
client.run(TOKEN)

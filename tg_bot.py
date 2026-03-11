import logging
import hashlib
import datetime

# from Load_model import Load_model
from resize import resize_image
from telegram import ForceReply, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

DOWNLOAD_FOLDER = "downloaded"


def getHash(text: str, hashLength: int):
    text_to_bytes = bytes(text, "utf-8")
    return hashlib.shake_256(text_to_bytes).hexdigest(hashLength // 2)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def saveImage(
    update: Update, context: ContextTypes.DEFAULT_TYPE, download_folder=DOWNLOAD_FOLDER
) -> None:
    # Get caption hash it, set filename
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    image_caption = update.message.caption
    hashed_caption = getHash(image_caption, 20)
    filename = f"{hashed_caption}_{date}"
    # Get and download photo
    image_file = await update.message.photo[-1].get_file()
    await image_file.download_to_drive(custom_path=f"{download_folder}/{filename}.jpg")
    resize_image(f"{filename}.jpg", "downloaded", "resized")
    await update.message.reply_text("Image saved!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    # CHANGEIT!!!
    application = Application.builder().token("TOKEN").build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    # handle photo
    application.add_handler(MessageHandler(filters.PHOTO, saveImage))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

import logging
import hashlib
import datetime
import torch
import torch.nn as nn
import os


from model import ConvNet
from PIL import Image
from torchvision import transforms
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
MODEL_PATH = "best_brain_model.pth"

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

model = ConvNet().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model initialized successfully")
except FileNotFoundError:
    print(f"Model's file {MODEL_PATH} not found")
except Exception as e:
    print(f"Error while initializing the model: {e}")
    model = None


def getHash(text: str, hashLength: int):
    text_to_bytes = bytes(text, "utf-8")
    return hashlib.shake_256(text_to_bytes).hexdigest(hashLength // 2)


def predict_tumor(image_path):
    if model is None:
        print("Model is not initialized")

    try:
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        pred_class = CLASSES[predicted.item()]
        confidence_score = confidence.item() * 100

        all_probs = {}
        for i, class_name in enumerate(CLASSES):
            all_probs[class_name] = probabilities[0][i].item() * 100

        return {
            "success": True,
            "class": pred_class,
            "confidence": confidence_score,
            "all_probabilities": all_probs,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def format_pred_result(result):
    if not result["success"]:
        return f"Error: {result['error']}"

    text = (
        f"Results:\n"
        f"{result['class']}\n"
        f"Confidence: {result['confidence']:.2f}\n"
        f"Important: This result is preliminary. Must be consulted with a specialist"
    )

    return text


async def handle_prediction(update: Update, image_path: str) -> None:
    processing_msg = await update.message.reply_text("Analizing image...")

    try:
        # Get prediction
        result = predict_tumor(image_path)
        # Format and send results
        response_text = format_pred_result(result)
        # Delete processing message
        # await processing_msg.delete()
        # Send results
        await update.message.reply_text(response_text)

    except Exception as e:
        await processing_msg.delete()
        await update.message.reply_text(f"Error while processing: {str(e)}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"File deleted: {image_path}")


async def saveImage(
    update: Update, context: ContextTypes.DEFAULT_TYPE, download_folder=DOWNLOAD_FOLDER
) -> None:

    # Get caption hash it, set filename
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    image_caption = update.message.caption
    hashed_caption = getHash(image_caption, 20)
    filename = f"{hashed_caption}_{date}"
    file_path = f"{DOWNLOAD_FOLDER}/{filename}.jpg"

    try:
        # Get and download photo
        image_file = await update.message.photo[-1].get_file()
        await image_file.download_to_drive(
            custom_path=f"{download_folder}/{filename}.jpg"
        )
        await update.message.reply_text("Image saved!")

        await handle_prediction(update, file_path)

    except Exception as e:
        await update.message.reply_text(f"Error while saving: {str(e)}")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


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


def main() -> None:
    """Start the bot."""
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    # Create the Application and pass it your bot's token.
    # CHANGEIT!!!
    application = (
        Application.builder()
        .token("TOKEN")
        .connect_timeout(120.0)
        .read_timeout(120.0)
        .write_timeout(120.0)
        .pool_timeout(120.0)
        .build()
    )

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

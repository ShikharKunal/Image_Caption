import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
from io import BytesIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from telegram import Update, Bot  
from telegram.ext import Updater, Application, CommandHandler, MessageHandler, filters, CallbackContext

from typing import Final
TOKEN: Final = "your-token-here"

STARTING, CAPTION = range(2)

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 74
img_size = 299
model = tf.keras.models.load_model("model.keras")
xmodel = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((img_size, img_size))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 127.5
    img = img - 1.0
    return img

# Function to map index to word using tokenizer
def idx_to_word(idx, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == idx:
            return word
    return None

# Function to generate caption
def generate_caption(img):
    img = preprocess_image(img)
    feature = xmodel.predict(img)

    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence])[0]
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    caption = in_text.split()[1:-1]
    caption = ' '.join(caption)
    return caption

# Telegram bot setup
async def start(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Send me an image to generate a caption!")
    return STARTING

async def handle_image(update: Update, context: CallbackContext):
    # Process the received image
    photo_file = await context.bot.get_file(update.message.photo[-1].file_id)

    # Download the file to memory
    photo_stream = BytesIO()
    await photo_file.download_to_memory(out=photo_stream)
    photo_stream.seek(0)

    # Open the image
    img = Image.open(photo_stream).convert("RGB")
    caption = generate_caption(img)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"{caption}")


def main():
    # Initialize the bot
    print('Starting bot...')
    app =  Application.builder().token(TOKEN).build()


    # Handlers
    start_handler = CommandHandler('start', start)
    image_handler = MessageHandler(filters.PHOTO, handle_image)

    # Add handlers to dispatcher
    app.add_handler(start_handler)
    app.add_handler(image_handler)

    # Start the Bot
    app.run_polling()
    print("Bot is running!")

if __name__ == '__main__':
    main()
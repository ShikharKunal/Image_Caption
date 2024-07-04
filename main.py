from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize FastAPI app
app = FastAPI()

# Initialize templates
templates = Jinja2Templates(directory="templates")

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

# Index route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Predict route
@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    caption = generate_caption(img)
    # Render result.html with file details and caption
    return templates.TemplateResponse("result.html", {"request": request,  "caption": caption})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
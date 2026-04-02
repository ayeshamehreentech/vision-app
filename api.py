import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model
model = tf.keras.models.load_model("retina_model.h5")

CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'GLAUCOMA', 'MH']

def preprocess(image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32")

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    img_array = preprocess(image)
    preds = model.predict(img_array)[0]

    return {
        "probabilities": preds.tolist()
    }
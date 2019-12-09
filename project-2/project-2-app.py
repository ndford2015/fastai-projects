from fastapi import FastAPI
from fastai.vision import *
from fastai.metrics import error_rate
from PIL import Image
import requests
from io import BytesIO

def get_bytes(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

app = FastAPI()
learn_path = Path('/home/jupyter/projects/project-2')
learner = load_learner(learn_path)

@app.get("/classify-food-url")
async def readUrl(url: str = ''):
    img = await get_bytes(url)
    _,_,losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(cat_learner.data.classes, losses),
            key=lambda p: p[1],
            reverse=True
            )
        })


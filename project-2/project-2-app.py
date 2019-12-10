from fastapi import FastAPI
from fastai.vision import *
from fastai.metrics import error_rate
from PIL import Image
import requests
from io import BytesIO
from starlette.responses import JSONResponse

def get_bytes(url):
    return requests.get(url)
    
 
app = FastAPI()
learn_path = Path('/home/jupyter/projects/project-2')
learner = load_learner(learn_path)

@app.get("/classify-food-url")
async def readUrl(url: str = ''):
    img_bytes = get_bytes(url)
    fastai_img = open_image(BytesIO(img_bytes.content))
    _,_,losses = learner.predict(fastai_img)
    return JSONResponse({
        "predictions": sorted(
            zip(learner.data.classes, map(float,losses)),
            key=lambda p: p[1],
            reverse=True
            )
        })


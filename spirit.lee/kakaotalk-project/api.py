# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi.responses import HTMLResponse
from dto import ChatbotRequest
from samples import simple_text_sample, basic_card_sample, commerce_card_sample
from callback import callback_handler
import vector_db
import os
from pathlib import Path
import glob
import threading

app = FastAPI()


## vector DB에 txt파일 embedding
def db_init() -> dict:
    files = glob.glob("./docs/*.txt")
    db_dict = dict()
    for name in files:
        print("creating vector db based on {}...".format(name))
        db_dict[Path(os.path.basename(name)).stem] = vector_db.init(name)
    return db_dict


app.docs = db_init()


@app.get("/")
async def home():
    page = """
    <html>
        <body>
            <h2>카카오 챗봇빌더 스킬 예제입니다 : )</h2>
        </body>
    </html>
    """
    return HTMLResponse(content=page, status_code=200)


@app.post("/skill/hello")
async def sample1(req: ChatbotRequest):
    return simple_text_sample


@app.post("/skill/basic-card")
async def sample2(req: ChatbotRequest):
    return basic_card_sample


@app.post("/skill/commerce-card")
async def sample3(req: ChatbotRequest):
    return commerce_card_sample


@app.post("/callback")
def callback1(req: ChatbotRequest, background_tasks: BackgroundTasks):
    print("callback handler executed: request payload: ", req)
    out = {
        "version": "2.0",
        "useCallback": True,
        "data": {
            "text": "생각하고 있는 중이에요😘 \n15초 정도 소요될 거 같아요 기다려 주실래요?!"
        }
    }
    background_tasks.add_task(callback_handler, req, app.docs)
    return out

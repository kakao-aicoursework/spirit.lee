# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi.responses import HTMLResponse
from dto import ChatbotRequest
from samples import simple_text_sample, basic_card_sample, commerce_card_sample
from callback import callback_handler
import vector_db
import threading

app = FastAPI()


## vector DB에 txt파일 embedding
def db_init():
    # files = ["project_data_카카오소셜.txt", "project_data_카카오싱크.txt", "project_data_카카오톡채널.txt"]
    files = ["project_data_카카오싱크.txt"]
    docs = vector_db.init(files)
    return docs


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
async def callback1(req: ChatbotRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(callback_handler, req, app.docs)
    print("callback handler executed: request payload: ", req)
    out = {
        "version": "2.0",
        "useCallback": True,
        "data": {
            "text": "생각하고 있는 중이에요😘 \n15초 정도 소요될 거 같아요 기다려 주실래요?!"
        }
    }
    return out

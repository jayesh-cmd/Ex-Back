import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic

app = FastAPI(title="Chat Analyser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mistake(BaseModel):
    title: str
    detail: str
    instead: str

class Message(BaseModel):
    label: str
    text: str

class Move(BaseModel):
    title: str
    detail: str

class Opener(BaseModel):
    idea: str
    text: str
    escalation: str

class AnalysisResponse(BaseModel):
    user_name: str
    her_name: str
    score: int
    score_reason: str
    mistakes: list[Mistake]
    messages: list[Message]
    moves: list[Move]
    openers: list[Opener]
    best_time: str
    best_time_reason: str

SYSTEM_PROMPT = """
You are a sharp, emotionally intelligent relationship coach with the energy of a gen-z best friend who actually knows what they're talking about. You're honest but not cruel. Flirty in your tone — like you're having fun with this — but never cheap or cringe. Think: the older sibling who's been through it, reads people well, and gives real advice not therapy-speak.

HOW TO DETECT WHO'S WHO:
- Look at the two names in the chat
- The person who uploaded this chat is the one seeking advice — they're usually the one who initiates more, has a shorter/nickname-style name, or sent the last message
- Make your best guess and state user_name and her_name clearly in the JSON

NOTE THE DATES:
- Look at the timestamps in the chat carefully
- Note when the last message was sent and how long ago that was relative to today
- Use this to make the messages section feel current and time-aware — not like a replay of the past

LANGUAGE RULES — FOLLOW EXACTLY:
- All analysis, advice, reasons, details → sharp English. Gen-z tone, no therapy-speak. Concise — 2-4 sentences max per field.
- mistakes[].instead → Hinglish. This is the reply they SHOULD HAVE sent at that exact moment in the chat — not advice for now, the literal alternative message they could have typed back then.
- messages[].text, openers[].text, openers[].escalation → Hinglish, matching the tone of how these two actually talk in the chat.

Analyse this WhatsApp chat export and return ONLY valid JSON, no extra text, no markdown, no backticks. Exactly this structure:

{
  "user_name": "detected name of the person seeking advice",
  "her_name": "detected name of the other person",
  "score": 0,
  "score_reason": "one sharp sentence on where things actually stand between them",
  "mistakes": [
    {
      "title": "short punchy mistake title",
      "detail": "specific, concise — 2-3 sentences max. quote actual messages from the chat. what happened and why it cost them.",
      "instead": "the exact hinglish reply they should have sent at that moment — not now-advice, the literal text they could have typed back then"
    },
    {
      "title": "second mistake title",
      "detail": "specific, concise — 2-3 sentences max. quote actual messages.",
      "instead": "the exact hinglish reply they should have sent at that moment in the chat"
    },
    {
      "title": "third mistake title",
      "detail": "specific, concise — 2-3 sentences max.",
      "instead": "the exact hinglish reply they should have sent at that moment in the chat"
    }
  ],
  "messages": [
    {
      "label": "opener",
      "text": "fresh hinglish message to send TODAY — time-aware, accounts for how long it has been since the last message, references something real from the chat but feels current not like a replay"
    },
    {
      "label": "agar wo warm ho",
      "text": "hinglish follow up — playful, pulls her in, references something she actually said in the chat"
    },
    {
      "label": "agar wo cold ho jaaye",
      "text": "short hinglish exit — unbothered, leaves door open"
    }
  ],
  "moves": [
    {
      "title": "short punchy move title",
      "detail": "direct, concise, 2-4 sentences, specific to this chat. what to do, what to avoid, why. like a smart friend talking not a coach writing a blog."
    },
    {
      "title": "second move title",
      "detail": "direct, concise, 2-4 sentences, specific to this chat."
    }
  ],
  "openers": [
    {
      "idea": "callback opener",
      "text": "exact hinglish message to send right now — pulled from something real in the chat, time-aware, confident, no desperation",
      "escalation": "english advice on the next move, then the actual hinglish line to send if she replies well"
    },
    {
      "idea": "fresh start opener",
      "text": "different angle, still grounded in who she is from this chat — hinglish",
      "escalation": "english advice on how to build from here, then the actual hinglish line to send"
    }
  ],
  "best_time": "HH:MM am/pm",
  "best_time_reason": "one sharp sentence on why — based on actual reply patterns visible in the chat"
}

Critical rules:
- mistakes[].instead = the reply they should have sent AT THAT MOMENT in the chat, in Hinglish. Not now-advice. The literal text they should have typed back then.
- messages[].text = fresh texts for TODAY. Time-aware. Not a replay of old conversation threads.
- moves[].detail = short, direct, specific to this chat. Like a smart friend talking. 2-4 sentences max.
- Be hyper-specific to THIS chat. Quote real messages and real dynamics from the actual conversation.
- Return ONLY the JSON object. Nothing else.
""".strip()

@app.get("/")
def serve_index():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(chat_file: UploadFile = File(...)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    try:
        raw = await chat_file.read()
        content = raw.decode("utf-8", errors="ignore")
        lines = [l for l in content.splitlines() if l.strip()]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read file: {str(e)}")

    if len(lines) < 10:
        raise HTTPException(status_code=400, detail="chat too short to analyse")

    trimmed = "\n".join(lines[-800:])

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyse this WhatsApp chat export:\n\n{trimmed}"
                }
            ]
        )
        raw_json = response.content[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"claude api error: {str(e)}")

    try:
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="ai response could not be parsed, try again")

    try:
        return AnalysisResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"response shape mismatch: {str(e)}")

"""The chat container's web app: a chat page over a finished pretraining leg,
named in the page's URL. The leg's manifest names the model and tokenizer, its
model.pt holds the weights, and replies stream back token by token. How the
prompt is built and rolled is docs/chat.md.
"""

import codecs
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from artifacts.core.artifact import Artifact
from config import STORAGE
from launcher.gate import gate
from launcher.state import volume

EOT = "<|endoftext|>"  # the document boundary every prompt starts from and every reply may end on
STOP = "\nUser:"  # the model starting the person's turn: the reply ends before it
MAX_NEW_TOKENS = 256
TOP_K = 50


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Conversation(BaseModel):
    """One POST /chat: the leg to answer with and the whole conversation so
    far, ending on the person's turn. The page holds it; the server keeps
    nothing between turns.

        Conversation(leg="runs/r/pretraining/0-4-b9b541d788", messages=[Message(role="user", content="Hi")])
        -> Conversation(leg='runs/r/pretraining/0-4-b9b541d788', system='', messages=[Message(role='user', content='Hi')], temperature=0.8)
    """

    leg: str
    system: str = ""
    messages: list[Message]
    temperature: float = Field(default=0.8, gt=0, le=2)


def build_chat(web_dir: Path) -> FastAPI:
    """Returns the chat page and its API, behind the dashboard's password.

    `web_dir` is where the image mounted `web/`, which holds the page,
    `chat.html`, and the login form it shares with the dashboard.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One leg on the GPU per container: asking for another replaces it.
    @lru_cache(maxsize=1)
    def load(leg_path: str):
        volume.reload()  # a leg finished since this container started is not in its view yet
        leg = Artifact.load(leg_path, STORAGE)
        model = leg.model.build(device)
        model.load_state_dict(torch.load(leg.paths(STORAGE)["model"], map_location=device)["model"])
        return leg, model.eval(), leg.tokenizer.bind(STORAGE)

    api = FastAPI()
    gate(api, web_dir / "login.html", public="/login")  # nothing here is public

    @api.get("/")
    def page() -> FileResponse:
        return FileResponse(web_dir / "chat.html")

    @api.get("/info")
    def info(leg: str) -> dict:
        loaded, _, _ = load(leg)
        return {"uid": loaded.uid, "sequence_length": loaded.model.sequence_length}

    @api.post("/chat")
    def reply(conversation: Conversation) -> StreamingResponse:
        leg, model, tokenizer = load(conversation.leg)
        [eot] = tokenizer.encode(EOT)
        context = leg.model.sequence_length
        # The start of a document and the system prompt stay in the window;
        # the transcript rolls through what is left, oldest tokens first out.
        # The system prompt takes at most half the window.
        system = conversation.system.strip()
        pinned = [eot, *tokenizer.encode(f"{system}\n\n" if system else "")][: context // 2]
        room = context - len(pinned)
        transcript = "".join(f"{m.role.capitalize()}: {m.content.strip()}\n" for m in conversation.messages)
        ids = tokenizer.encode(transcript + "Assistant:")
        dropped = max(0, len(ids) - room)

        def stream():
            # A character can span tokens, so bytes go through an incremental
            # decoder; and the tail that could still grow into STOP is held
            # back until the next token says whether it does.
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            text, sent = "", 0
            with torch.inference_mode():
                for _ in range(MAX_NEW_TOKENS):
                    window = torch.tensor([pinned + ids[-room:]], device=device)
                    top = torch.topk(model(window)[0, -1].float(), TOP_K)
                    probs = torch.softmax(top.values / conversation.temperature, dim=-1)
                    token = top.indices[torch.multinomial(probs, 1)].item()
                    if token == eot:
                        break
                    ids.append(token)
                    text += decoder.decode(tokenizer.vocab[token])
                    if STOP in text:
                        yield text[sent : text.index(STOP)]
                        return
                    held = max(k for k in range(len(STOP)) if text.endswith(STOP[:k]))
                    yield text[sent : len(text) - held]
                    sent = len(text) - held
            yield text[sent:] + decoder.decode(b"", final=True)

        # Server-sent events, which proxies pass on as they come rather than
        # buffering; each piece is JSON so a newline in it cannot end the event.
        return StreamingResponse(
            (f"data: {json.dumps(piece)}\n\n" for piece in stream() if piece),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Window-Dropped": str(dropped)},
        )

    return api

# Chat

A chat page over a finished pretraining leg: `main.py`'s `chat` function, on
its own A10, serving `launcher.chat` and the page `web/chat.html`. It reads the
volume and writes nothing, and takes no lease.

The leg is named in the page's URL, `/?leg=<artifact_path>`. A done
`Pretraining` row on the dashboard links there through its **serve** button,
`GET /serve/<path>` on the dashboard, which redirects to the chat function's
URL ([Dashboard](dashboard.md#row-buttons)), and the visit starts its
container. The page sits behind the dashboard's password, and the redirect
carries a ticket that logs the browser in on the chat's own domain, so the
password is not asked for again ([Dashboard](dashboard.md#access)).

## Loading a leg

A leg's manifest already says everything the weights alone do not: its `model`
names the architecture and its shapes ([Models](models.md)), and its dataset
names the tokenizer. So the first request naming a leg reloads the volume (a
leg finished since the container started is not in its view yet), loads the
leg, builds `leg.model` on the GPU, loads the `"model"` state_dict out of its
`model.pt` ([Training](training.md)), and binds `leg.tokenizer`. Each reply
runs under the leg's own `autocast`, in the precision the leg trained in. The
page asks for the leg as it opens, so a cold start shows there rather than on
the first reply. A container holds one leg; a request naming another replaces it. A leg
whose `model.pt` is not written yet fails at the load, naming the missing file.

## The prompt

The models here are pretrained on plain text, with no chat template and no
role tokens, so a conversation is written as text the model can continue:

```text
<|endoftext|>{system}

User: {message}
Assistant: {reply}
User: {message}
Assistant:
```

The reply ends on `<|endoftext|>`, on `"\nUser:"` (the model starting the
person's turn, which is cut off), or after 256 tokens. Sampling is top-k (50)
at the temperature the page sends.

## The rolling window

The page holds the conversation and sends all of it every turn; the server
keeps nothing between turns. What reaches the model is at most
`sequence_length` tokens: the `<|endoftext|>` and the system prompt are
pinned at the front (taking at most half the window), and the transcript
fills the rest from its newest token back. The oldest tokens fall off first,
mid-turn if that is where the cut lands, and the window keeps sliding while
the reply is generated. How many transcript tokens fell off comes back in the
`X-Window-Dropped` header, and the page marks the conversation when it is not
zero.

Each token re-runs the model over the whole window, as the models keep no
cache between steps.

## Streaming

`POST /chat` answers with server-sent events (`text/event-stream`), one
`data:` event per token, its text JSON-encoded so a newline in it cannot end
the event. A token's bytes go through an incremental UTF-8 decoder, so a
character split across tokens is sent once it is whole, and any tail that
could still grow into `"\nUser:"` is held back until the next token decides
it. The page shows the reply a word at a time: a word appears once the
whitespace after it arrives, and lines wrap between words. Stopping on the
page aborts the request, and generation ends at the next token.

## Why

**An A10.** A `TransformerFA` leg's attention runs only on flash attention,
which takes fp16 or bf16 on an Ampere GPU or later. The A10 is an Ampere GPU,
so it serves every leg, in bf16 under the leg's `autocast`.

**A leg folder, not a `.pt` path.** A `model.pt` holds tensors and nothing
about the network they belong to or the tokenizer they speak. Pointing at the
leg reads both from its manifest, so they cannot drift from the weights.

**Its own container, not a job.** Serving a model produces no artifact and
holds the GPU for as long as someone is talking to it, so it takes no lease
and the launcher never sees it. It scales down after five idle minutes, as any
web function does.

**Pinned system prompt.** The system prompt is what the person set the chat
up with; letting it roll out would change the chat's behaviour without the
person changing anything.

## Reference

::: launcher.chat

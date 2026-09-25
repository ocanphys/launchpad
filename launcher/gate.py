"""The password in front of the dashboard.

One password over the whole app: it launches and cancels jobs, and its `/lab`
route redirects with the lab's token attached, so an open URL there is an open
shell on the volume. DASHBOARD_PASSWORD is the one a person types and picks;
JUPYTER_TOKEN stays whatever random string the secret was made with, because
it travels in a URL.

The cookie is the password's own digest: no session store, nothing to expire,
and a container that restarts leaves every browser that had logged in still
logged in.
"""

import hashlib
import hmac
import os
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

COOKIE = "launchpad"


def gate(api: FastAPI, login_html: Path) -> None:
    """Puts every route on `api` but `/login` behind DASHBOARD_PASSWORD.

    Call it after any middleware that should stay outside the password (an
    error logger), since Starlette wraps in reverse registration order, and
    before serving anything.
    """
    session = hashlib.sha256(os.environ["DASHBOARD_PASSWORD"].encode()).hexdigest()
    form = login_html.read_text()

    @api.middleware("http")
    async def require_password(request, call_next):
        # Bytes, not str: a cookie is whatever was sent, and `compare_digest`
        # refuses a str that isn't ASCII rather than saying no to it.
        cookie = request.cookies.get(COOKIE, "").encode("utf-8", "replace")
        if request.url.path == "/login" or hmac.compare_digest(cookie, session.encode()):
            return await call_next(request)
        # A browser navigating gets the form; a fetch from the page gets a
        # status it can report, not a login page parsed as JSON.
        if request.method == "GET":
            return RedirectResponse("/login", status_code=303)
        return Response(status_code=401)

    @api.get("/login")
    def login_page() -> HTMLResponse:
        return HTMLResponse(form)

    # The body is read rather than declared as a `Form(...)` field: form
    # parsing pulls in python-multipart, and one urlencoded field out of
    # `parse_qs` is the whole of what that dependency would do here.
    @api.post("/login")
    async def login(request: Request) -> Response:
        given = parse_qs((await request.body()).decode()).get("password", [""])[0]
        if not hmac.compare_digest(hashlib.sha256(given.encode()).hexdigest(), session):
            return HTMLResponse(form.replace("<!--note-->", "wrong password"), status_code=401)
        answer = RedirectResponse("/", status_code=303)
        answer.set_cookie(COOKIE, session, max_age=30 * 86400, httponly=True, secure=True, samesite="lax")
        return answer

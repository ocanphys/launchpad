"""The password in front of the dashboard.

One password over the whole app: it launches and cancels jobs, and its `/lab`
route redirects with the lab's token attached, so an open URL there is an open
shell on the volume. The one exception is the docs site, which is public.
DASHBOARD_PASSWORD is the one a person types and picks;
JUPYTER_TOKEN stays whatever random string the secret was made with, because
it travels in a URL.

The cookie is the password's own digest: no session store, nothing to expire,
and a container that restarts leaves every browser that had logged in still
logged in.
"""

import hashlib
import hmac
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

COOKIE = "launchpad"
TICKET = "ticket"  # the query parameter a ticket() travels in
TICKET_SECONDS = 60  # how long a ticket logs a browser in for: long enough for one redirect


def _session() -> str:
    return hashlib.sha256(os.environ["DASHBOARD_PASSWORD"].encode()).hexdigest()


def _sign(session: str, expires: str) -> str:
    return hmac.new(session.encode(), expires.encode(), "sha256").hexdigest()


def ticket() -> str:
    """Returns a login good for TICKET_SECONDS on any app behind this gate: what
    a route already past the password adds to a link to another domain, as
    `?ticket=`, so the browser is not asked for the password again there.

        ticket() -> "1793251476.b24f386730f5dc8c9637402044e72b70baf37967dd80fbfd9bc4fad8d678eace"
    """
    expires = str(int(time.time()) + TICKET_SECONDS)
    return f"{expires}.{_sign(_session(), expires)}"


def gate(api: FastAPI, login_html: Path, public: str) -> None:
    """Puts every route on `api` behind DASHBOARD_PASSWORD but `/login`, and
    `public` and everything under it. A GET carrying a live `ticket()` is
    logged in and sent on to the same URL without it.

    Call it after any middleware that should stay outside the password (an
    error logger), since Starlette wraps in reverse registration order, and
    before serving anything.
    """
    session = _session()
    form = login_html.read_text()

    def logged_in(answer: Response) -> Response:
        answer.set_cookie(COOKIE, session, max_age=30 * 86400, httponly=True, secure=True, samesite="lax")
        return answer

    @api.middleware("http")
    async def require_password(request, call_next):
        # Bytes, not str: a cookie is whatever was sent, and `compare_digest`
        # refuses a str that isn't ASCII rather than saying no to it.
        cookie = request.cookies.get(COOKIE, "").encode("utf-8", "replace")
        path = request.url.path
        open_path = path in ("/login", public) or path.startswith(public + "/")
        if open_path or hmac.compare_digest(cookie, session.encode()):
            return await call_next(request)
        expires, _, signature = request.query_params.get(TICKET, "").partition(".")
        if (
            request.method == "GET"
            and expires.isdigit()
            and int(expires) > time.time()
            and hmac.compare_digest(signature.encode("utf-8", "replace"), _sign(session, expires).encode())
        ):
            rest = urlencode([(key, value) for key, value in request.query_params.multi_items() if key != TICKET])
            return logged_in(RedirectResponse(f"{path}?{rest}" if rest else path, status_code=303))
        # A browser navigating gets the form, which brings it back here after;
        # a fetch from the page gets a status it can report, not a login page
        # parsed as JSON.
        if request.method == "GET":
            target = f"{path}?{request.url.query}" if request.url.query else path
            return RedirectResponse(f"/login?{urlencode({'next': target})}", status_code=303)
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
        # Only a path on this site: a `next` naming another host ("//evil",
        # which a browser also reads from "/\evil") would make the form a way
        # to send someone anywhere with this site's URL.
        target = request.query_params.get("next", "/")
        if not target.startswith("/") or target.startswith(("//", "/\\")):
            target = "/"
        return logged_in(RedirectResponse(target, status_code=303))

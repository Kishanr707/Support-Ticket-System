# Frontend Integration Guide

This backend is a FastAPI server that predicts a support ticket's priority
(`Blocker`, `Highest`, `High`, `Medium`, or `Low`) from its text. Your
frontend calls one endpoint and gets a prediction back as JSON — no need
to touch any Python or ML code.

## Running the backend locally

Right now this only runs on `localhost`, so to integrate and test against
it, clone the backend repo and run it alongside your frontend:

```bash
git clone <backend-repo-url>
cd "Support Ticket System"
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

Once running, the API is live at `http://127.0.0.1:8000`. You can browse
`http://127.0.0.1:8000/docs` for interactive API docs (auto-generated,
lets you try requests right in the browser).

## The endpoint

**`POST /predict`**

Request body:
```json
{ "ticket_text": "The VPN keeps disconnecting every few minutes" }
```

Response body:
```json
{ "priority": "High" }
```

`priority` will always be one of: `"Blocker"`, `"Highest"`, `"High"`,
`"Medium"`, `"Low"`.

If `ticket_text` is empty or missing, the API returns a `422` validation
error — worth handling in the UI (e.g. disable submit until there's text).

## Example call (JavaScript / fetch)

```javascript
async function getPriority(ticketText) {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticket_text: ticketText }),
  });

  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.status}`);
  }

  const data = await response.json();
  return data.priority; // e.g. "High"
}
```

CORS is already enabled on the backend, so calls from a React dev server
(e.g. `localhost:3000`) won't be blocked.

## Health check

**`GET /health`** — returns `{"status": "ok"}` if the server is up. Useful
for a quick "is the backend running" check before debugging anything else.

## Known limitations right now — please read before demoing

- **No authentication yet.** Anyone who can reach the URL can call
  `/predict`. Don't rely on this for anything sensitive yet.
- **No database.** The API only answers one prediction at a time — it
  doesn't store ticket history, so there's nothing to list/browse yet.
  If your UI needs a ticket list, that's not available from this backend
  yet — flag it and we'll prioritize the database work.
- **Accuracy is uneven across priority levels.** The model is currently
  strong on `Medium`/`High` (~87-88% correct) but weak on `Blocker`/`Low`
  (currently near 0% — not yet reliably learned). Worth knowing so a demo
  doesn't get derailed by an unlucky example — pick a clearly
  routine or clearly severe-sounding ticket for demos, and expect
  Blocker/Low examples to be hit-or-miss for now.
- **Only runs locally for now.** Nothing is deployed publicly yet — both
  of you need to be running this on your own machines (or the same
  network) to integrate and test together.

## Questions / issues

If a request isn't behaving as expected, check `/docs` first to confirm
the request shape is correct, then flag it — happy to adjust the API
contract if something doesn't fit how the frontend needs to use it.

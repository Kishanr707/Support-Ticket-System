# Support Ticket System — Roadmap

_Last updated: after completing the ML baseline + FastAPI integration stage._

## Where we are right now

The core ML pipeline and API are built, tested, and working:

- Real dataset (357 tickets) cleaned and labeled by priority
- TF-IDF + LogisticRegression baseline model, trained and evaluated honestly
- Combined with a ChatGPT-generated synthetic dataset (2,855 rows) — improved real-world macro-F1 from 0.20 to 0.38
- Wrapped in a stable `predict(ticket_text) -> priority` interface
- Exposed via a FastAPI `/predict` endpoint, tested, CORS-enabled, ready for frontend integration
- CI (GitHub Actions) runs the full test suite automatically on every push

## Current model performance (honest numbers)

Overall accuracy: **71%** — but this is not evenly distributed across priority levels.

| Priority | Correctly identified (recall) |
|---|---|
| Medium | 87% |
| High | 88% |
| Highest | 20% |
| Blocker | 0% |
| Low | 0% |

The model is strong on the three most common priority levels and has not yet learned to reliably identify `Blocker` or `Low` tickets, due to too few labeled examples of those classes. This is a data problem, not a code problem, and is the top priority for the next data-collection push.

## Remaining phases

| Phase | Task | Rough effort | Depends on |
|---|---|---|---|
| 1 | Lightweight DevSecOps (Dependabot, secret scanning, dependency audit in CI) | ~1 hour | Nothing |
| 2 | Docker packaging | 1 focused session | Nothing |
| 3 | Targeted data collection for Blocker/Low, retrain | A few days–2 weeks | Team writing realistic examples |
| 4 | Database setup (schema, wire into FastAPI) | A few days | Nothing |
| 5 | Feedback/correction table + admin dashboard (human confirms/corrects predicted priority) | ~1 week | Phase 4 |
| 6 | Role-based access control + real authentication | A few days | Can run in parallel with Phase 5 |
| 7 | Retraining pipeline using verified corrections | A few days setup | Phase 5 |
| 8 | Hosting / cloud deployment | ~3–5 days | Phase 2 |
| 9 | Kubernetes | Adds meaningful time beyond Phase 8 — worth deciding deliberately whether this is needed, vs. a simpler single-container cloud deploy | Phase 8 |
| — | Frontend integration | Already unblocked | Nothing — ready now |

**Suggested order:** Phases 1–2 first (quick, low-risk), then split effort — data collection (Phase 3) and database setup (Phase 4) can happen in parallel since they don't depend on each other.

## Design principles to keep in mind going forward

- **Always evaluate on real data, never trust synthetic/augmented data's own performance numbers at face value** — they tend to look better than they are.
- **Retrain only on human-confirmed priorities**, never on the model's own predictions — otherwise mistakes reinforce themselves instead of getting corrected.
- **`predict(ticket_text)` is the stable interface** — the model behind it can change freely without breaking the API or frontend, as long as this function's signature stays the same.

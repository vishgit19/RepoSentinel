# demo-auth-service

A miniature authentication service used as a RepoSentinel benchmark fixture.

## Layout

- `app/auth/token.py` — signed session tokens (`SessionToken`, `issue_token`, `decode_token`)
- `app/auth/middleware.py` — `validate_token` / `authenticate_request` used by the HTTP layer
- `app/auth/session.py` — session creation and revocation bookkeeping
- `app/users/repository.py` — in-memory user store
- `app/api.py` — dependency-free routing façade

## Contract

A session token is valid from `issued_at` until `expires_at`. Once
`expires_at` has passed the token must be rejected with `token_expired`.

## Running the tests

```bash
pytest -q
```

# user-directory

A tiny user directory with an email lookup.

## Layout

    app/db/engine.py           SQL evaluator (supports ``?`` placeholders)
    app/users/repository.py    user lookup issued as SQL
    app/api.py                 GET /users?email=...

Lookups must treat the email as data. The engine already supports bound
parameters; the repository is expected to use them.

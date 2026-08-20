# Security

RepoSentinel runs model-generated commands against a sandboxed copy of a
repository. Treat a running instance as a privileged local tool: do not point
it at secrets you would not give a developer with shell access to that machine.

## Reporting a vulnerability

Open a private GitHub security advisory on this repository, or email the
maintainer through the address on their GitHub profile. Please do not file a
public issue for an unfixed vulnerability.

## What this project already refuses

- Path escape out of the workspace
- Commands such as `rm`, `curl`, `pip`, and `python -c` in the local sandbox
- Opening a GitHub pull request unless a human approved the run **and**
  `REPOSENTINEL_ALLOW_GITHUB_PUSH` is set
- Exposing `apply_patch` / `write_file` on the MCP server

# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | Yes                |
| 0.2.x   | Yes                |
| 0.1.x   | No                 |

## Reporting a Vulnerability

Report a vulnerability privately through GitHub's private vulnerability reporting, as the
[security policy of the organization][org-security] describes. Open the
[Security tab](https://github.com/cuju-football/engine/security) of the repository, then choose
Report a vulnerability. Please keep vulnerabilities out of public issues and discussions.

A useful report names the version, the platform and the backend, and gives the steps or the file
that reproduce the problem. The organization's policy gives what to expect after a report.

## Trusted Input

Several of cuju's inputs are code or pickles. Load only the ones that come from trusted sources.

- **Scenario files are Python.** A batch or `cuju.scenarios.build_bundle` that takes a scenario file
    runs it with the permissions of the caller.
- **States are pickles.** `set_state` reads a state through a restricted unpickler that accepts only
    GRF's episode class and plain values, and for a stacked state the numpy arrays of its frames.
    It still parses bytes from outside.
- **Dumps are pickles too.** `cuju.dumps.read_dump` reads them through a restricted unpickler that
    accepts only the classes of GRF's dumps.
- **Action logs name their scenarios.** `cuju.replay` creates the batch that a log describes, so a
    log that names a scenario file runs that file.

## The Renderer

The renderer, the package cuju-render, plays submitted code and serves an API on the
loopback interface. Its own [security policy][render-security] describes both and takes their
reports.

[render-security]: https://github.com/cuju-football/render/blob/main/SECURITY.md
[org-security]: https://github.com/cuju-football/.github/blob/main/SECURITY.md

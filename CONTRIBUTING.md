# Contributing

This is an independent community project, not an official AzuraCast component.
Keep changes focused and describe the user-visible behavior in the pull request.

## Development

Use Linux and Python 3.11 or newer. Install the project and run its tests:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
cd tests
python3 -m unittest discover -s .
```

Installer ownership tests also need Linux root and Git; run them in a disposable
container or VM, not against a station's production installation. Tests use neutral
fixtures and do not require real API credentials, audio or a GPU.

Do not commit credentials, configuration, audio, databases, reports, station or
track identities, or listening annotations. The frozen selector dataset contains
derived numeric features and labels only. Preserve upstream license files verbatim.

Pod source changes can invalidate analysis caches. Detection changes need controlled
comparison against stored evidence before replacing published results; decision
changes need a dry-run comparison. Do not run paid GPU work without authorization.
Never overwrite intro values that were not written by the detector.

## Bug Reports

Include the installed version, Linux/Docker versions, the command used, expected
behavior and a small redacted error excerpt. Remove API keys, registry credentials,
station addresses, track names, audio paths and private analysis data.

Report security vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
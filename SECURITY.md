# Security

Report vulnerabilities through GitHub's private vulnerability reporting:
open **Security > Report a vulnerability** in this repository. Do not open a public
issue containing an exploit, credentials, station data, audio or private reports.

Provide the affected release, the vulnerable code path, a minimal reproduction
using neutral data and the expected impact. Never send production API keys.
If a key may have been exposed, revoke or rotate it at its provider.

Keep the detector updated. Pre-releases are experimental; no fixed support period
or response-time guarantee is offered. Do not use a pre-release to publish intro
values without reviewing its results first.

Install the detector through the Sweepers plugin: it gets its own AzuraCast user with
station View and Media permissions on the stations that use it, and nothing is analyzed
or written until a station switches it on. The AzuraCast key stays in the private `.env`
file, Runpod keys in the data volume (mode 600); never put keys in source code, issue
reports or command arguments.
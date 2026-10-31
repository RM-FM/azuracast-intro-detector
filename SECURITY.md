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

Use a dedicated AzuraCast API user with station View and Media permissions only.
Store keys in the private `.env` file, never in source code, issue reports or command
arguments. First installation stays in shadow mode with manual processing.
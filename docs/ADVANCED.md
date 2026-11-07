# Advanced guide

The detector is normally installed, updated and configured through the
[Sweepers plugin](https://github.com/RM-FM/azuracast-sweepers). This guide covers what
happens behind it and the cases it does not handle. Commands below run from the detector
checkout. Use a root shell or prefix Docker and file-administration commands with `sudo`.

## Requirements

AzuraCast with the Sweepers plugin on a Linux x86_64 server with Docker Compose v2.20 or
newer, Git and root access. The AzuraCast **Site Base URL** (Administration → System
Settings) must be set.

## Installation and updates through Sweepers

The Sweepers plugin's setup offers the detector when the plugin is installed. To add it
later, run in the plugin directory:

```bash
cd /var/azuracast/plugins/azuracast-sweepers
sudo bash bin/setup.sh --detector
```

The detector is installed to `/opt/azuracast-intro-detector`, gets its own AzuraCast
account and joins AzuraCast's Docker network. It opens no port and does not restart
AzuraCast or interrupt broadcasting. No API key is entered on the command line.

It is updated with the plugin: `sudo bash bin/setup.sh` in the plugin directory moves it
to the release that plugin version expects. Stored analyses, station settings and intro
values are kept.

Logs of the running service:

```bash
cd /opt/azuracast-intro-detector
sudo docker compose logs --tail 50 detector
```

## Runpod API key

1. Sign in to [Runpod](https://console.runpod.io/) and add funds to the account.
2. Open [Credentials](https://console.runpod.io/user/credentials), select **API Keys**
   and click **Create API Key**.
3. Name it for the detector. Give it read/write access for Pod and GPU management;
   a **Read Only** key cannot create or stop workers. You do not need an S3 key.
4. Keep the key in a password manager; it is entered only on the Intro Detector page.

## Installation and Compose

The detector is independent of AzuraCast's containers. It needs outbound HTTPS to
AzuraCast and Runpod and SSH/rsync access to its Runpod workers. Its management API
listens only on AzuraCast's Docker network; no port is published on the host.
Install Docker Engine, the Compose plugin and Git before running [setup](../bin/setup.sh).
Compose v2.20+ is needed for the setup script's readiness wait.

The checkout contains:

| Path | Purpose |
| --- | --- |
| `config.toml` | Operator settings and defaults; no API keys |
| `.env` | AzuraCast API key and management API token; private, never committed |
| `compose.override.yaml` | Joins AzuraCast's Docker network; written by setup |
| `secrets/runpod-ssh-key` | Worker SSH private key; owned by UID 10001, mode 600 |
| `.setup-backups/` | Private copies of settings before reconfiguration |
| `detector-data` Compose volume | Database, analyses, queue, decisions, reports and Runpod keys |

The Sweepers plugin's setup runs `bin/setup.sh --azuracast-dir DIR --unattended` and
passes the AzuraCast site address and a new key of its detector account on standard
input. Setup then:

- builds the service image and checks the repository-provided public worker image
  using registry metadata, without downloading it or starting a Pod;
- writes `config.toml` and `.env`, creating the management API token once;
- joins the network of AzuraCast's `web` container under the alias `intro-detector`;
- runs `check`, starts the service and writes `.detector.json` into the plugin
  directory, readable by AzuraCast's web user only.

It does not run a backfill, enable a station or change AzuraCast broadcasting. Empty
input lines keep the existing address and key. Without `--unattended` setup asks for
them interactively. Credential checks must succeed before the detector is restarted.
If setup fails, its message gives the private technical log and settings-backup location.

`bin/update.sh --version vX.Y.Z --azuracast-dir DIR` moves the checkout to a release
tag and reruns setup unattended. If setup fails, the previous release and image run
again. The plugin's setup calls it with the release in its `detector.version`.

Every setup cleans up after itself: it removes the replaced service image, keeps the five
newest `.setup-backups`, and limits Docker's build cache to 1 GB. Docker manages that cache
for the whole host; AzuraCast pulls its images and builds none. A successful update also
removes the `azuracast-intro-detector:previous` rollback image.

The detector reaches AzuraCast through the public site address. Private-network
HTTP is supported only with `[azuracast] allow_http = true`; AzuraCast redirects plain
HTTP when "Always Use HTTPS" is on. Custom `COMPOSE_FILE` setups require manual installation.

## Manual installation

Clone a release next to AzuraCast and prepare the files:

```bash
git clone --branch vX.Y.Z https://github.com/RM-FM/azuracast-intro-detector.git /opt/azuracast-intro-detector
cd /opt/azuracast-intro-detector
cp config.example.toml config.toml
cp .env.example .env
nano config.toml
nano .env
chmod 600 .env
chown root:10001 config.toml
chmod 640 config.toml
mkdir -p secrets
chmod 700 secrets
ssh-keygen -t ed25519 -N '' -f secrets/runpod-ssh-key -C azuracast-intro-detector
chown 10001:10001 secrets/runpod-ssh-key
chmod 600 secrets/runpod-ssh-key
docker compose build detector
docker compose run --rm --no-deps detector check
docker compose up -d --no-deps --wait detector
```

Set the HTTPS address in `config.toml` and the AzuraCast API key and a random
`INTRO_DETECTOR_API_TOKEN` of at least 32 characters in `.env`. The API key's user needs
station **View** and **Media** on each station the detector should process. Add a
`compose.override.yaml` that joins AzuraCast's network with the alias `intro-detector`
(see setup) and write the plugin's `.detector.json` with `{"url": "http://intro-detector:8080",
"token": "…"}`. Standard deployments use the repository's default worker image.
Do not regenerate an existing worker key without accounting for any active worker.

## Manual update

Plugin-managed installations are updated by the plugin's setup. Otherwise allow active
scans to finish, read the changes and back up configuration and data, then:

```bash
sudo bash bin/update.sh --version vX.Y.Z --azuracast-dir /var/azuracast < /dev/null
```

or, for a checkout without the plugin:

```bash
git pull --ff-only origin main
docker compose build detector
docker compose run --rm --no-deps detector check
docker compose up -d --no-deps --force-recreate --wait detector
docker compose exec -T detector intro-detector status
```

The Compose volume is retained. Do not use `down --volumes` for an update.
Automatic processing resumes according to the station settings after restart.
Worker-image changes are separate from service updates and do not require
reanalyzing unchanged audio automatically.

The first start of 1.0 moves an existing configuration into the station settings:
stations selected in `config.toml` stay enabled with their filters and each
gets the Runpod key from `.env`. Later changes to station sections in `config.toml`
have no effect.

## Configuration reference

[config.example.toml](../config.example.toml) is ordered from ordinary settings to
advanced calibration. Omitted fields use the service defaults. Recreate the detector
after changing configuration so file mounts also pick up atomic editor replacements:

```bash
docker compose up -d --no-deps --force-recreate --wait detector
```

### Stations and filters

Stations are enabled and configured on the Sweepers plugin's **Intro Detector** page,
or with `intro-detector station set`. Settings per station:

| Setting | Default for a new station | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Sync, analyze and decide for this station |
| `auto_process` | `[scan] auto_process` | Analyze new uploads without "Process now" |
| `min_song_seconds` | `[station_defaults]`, `60` | Skip shorter files and files of unknown length |
| `max_song_seconds` | `900` | Skip longer files, such as DJ mixes; `0` means no maximum |
| `exclude_playlists` | `[station_defaults]`, `[]` | Additional playlist IDs excluded from analysis |
| `budget_usd` | `[runpod] monthly_budget_usd`, `10` | Monthly GPU budget of the station; `0` means none |
| `idle_timeout_minutes` | `5` | Minutes the GPU worker stays after its last batch (1 to 60) |

Each station also has its own Runpod key (`intro-detector key set <station>`).
`[azuracast] stations` and `[[stations]]` blocks are read only once, when an existing
installation is migrated.

Jingle Mode playlists are always excluded. The detector's AzuraCast account gets
station **View** and **Media** from the plugin when a station is enabled. Never give the
detector a global administrator key merely to resolve missing station permissions.

### Processing

| `[scan]` setting | Service default | Meaning |
| --- | --- | --- |
| `auto_process` | `true` | Default of a newly enabled station |
| `poll_seconds` | `300` | Library discovery interval |
| `batch_quiet_seconds` | `120` | Wait after the last new upload before starting a batch |
| `batch_max_songs` | `100` | Maximum originals per batch |
| `max_attempts` | `2` | Retry limit for failures |

First discovery records the existing library without queuing it all. Later, only new
media IDs with a blank `intro_end` are queued automatically; changes to existing media
are recorded but not automatically reanalyzed. `backfill` queues existing blank intros
at lower priority. "Process now" (`process --station`) skips the quiet period for one
station, also when automatic processing is off.

A batch contains the songs of stations on the same Runpod account. Songs of a station
without key, with an exhausted budget or after a failed batch on its account wait; they are
downloaded once to learn their audio hash, so an analysis stored for another station
is still reused.

### Decision settings

| `[decision]` setting | Default | Meaning |
| --- | --- | --- |
| `no_intro_below_seconds` | `2.0` | Earlier vocals leave no usable intro and map to zero |
| `music_min_seconds` | `3.0` | Minimum musical fallback cue |
| `music_max_seconds` | `27.0` | Maximum musical fallback cue |

`[decision.vocals]` contains activity, burst-duration and ad-lib calibration. Change
these only with an A/B comparison and listening validation. They affect stored
evidence without requiring another GPU analysis. Non-default settings add a hashed
configuration suffix to each decision policy.

Model inference parameters and frequency analysis belong to the worker code/image,
not this configuration. Changing them can require fresh analyses.

## Publishing and automatic processing

An enabled station writes every intro the detector finds to AzuraCast at once; values
entered by hand are kept. Start with one station. After changing decision settings,
check and apply their effect on stored analyses from the command line:

```bash
docker compose exec detector intro-detector decision-diff --station 1
docker compose exec detector intro-detector republish --station 1
```

The dry-run report lists proposed changes. Republishing uses stored analyses and does
not create a GPU worker. Check results before applying them.

Switch off **Analyze new uploads automatically** when uploads should wait for "Process
now".

Earlier releases had a shadow mode that only recorded intros. Updating writes the intros
recorded that way once for every enabled station.

## Runpod keys and the management API

Every station enters a Runpod key and a monthly budget. Several stations may enter the
same key or each use its own. Stations whose keys belong to the same Runpod account
(Runpod reports the account of a key when it is saved, otherwise within ten minutes)
share one worker (`azuracast-intro-detector-` and eight characters of the hashed
account ID), lock and balance. Each worker's cost, including its start and idle time,
is split between its stations by analyzed songs, and each station's share counts
against its own budget. A worker that is running when its key's account becomes known
keeps its records until it is retired. Keys are checked with Runpod before they
are stored in `secrets.json` (mode 600) in the data volume and reach `runpodctl` only
through its environment. A station cannot remove or replace a key while a worker runs
on it and no other station uses that key.

Stored analyses are reused for identical files across stations; a file is analyzed and
paid for only once.

The service starts the management API on port 8080 when `INTRO_DETECTOR_API_TOKEN` is
set. Every request needs that bearer token; the API never returns keys. It serves
`/api/v1/health`, `/api/v1/stations` (IDs and whether enabled) and `/api/v1/stations/{id}` (settings, key, status, process, retry,
backfill, publish preview, publish and worker retirement). A station's status includes
the number of stations using its key, their combined spend this month and the key's
Runpod balance (checked every ten minutes). The Sweepers plugin enforces AzuraCast
permissions before calling the API; do not expose the port elsewhere.

## Automatic GPU workers

The repository supplies the default worker-image reference and is responsible for
preparing, testing and publishing it before an operator release. Stations and
station managers provide Runpod keys on the Intro Detector page; they do not build worker
images, choose CUDA packages or configure registry credentials for the standard installation.

Setup verifies that the default public image is accessible anonymously before changing
credentials or starting the service. If it is unavailable, setup stops with a
maintainer-release error. Do not work around that failure by choosing an arbitrary
image. Full model/GPU compatibility is a maintainer validation responsibility,
not something registry metadata alone can establish.

New workers run in Secure Cloud with one GPU and no network volume. Originals and
runtime code are transferred by SSH/rsync; only reports are copied back. The worker
receives an SSH public key, not the AzuraCast API key.

Custom worker images are an expert override, not part of normal setup. Existing
private-image configuration is preserved and skips the anonymous public-image check.
Its maintainer is responsible for compatibility and Runpod registry credentials.

## Costs and limits

The README's **USD 2 to 6 per 1,000 new songs** is a planning range for prepared
RTX 4090 workers and batched processing, not a measured guarantee for every workload.
The GPU ranking defaults use 15 seconds per song for one lane and five minutes of
startup; three overlapping lanes can reduce wall time. Cold startup and the idle timeout
matter most for small batches. Storage, retries, image
downloads, provider billing and changing rates can add cost.

Previously analyzed audio is reused by SHA256 across stations without GPU inference.
The detector host's own hosting and storage costs are separate.

Budgets are set per station on the Intro Detector page. `[runpod]` holds the initial
budget of a newly enabled station and the hourly price cap of all workers:

```toml
monthly_budget_usd = 10.0
max_price_per_hour = 0.90
```

Each station budget uses estimated spend recorded by this detector for a UTC calendar
month. `0` disables it. It keeps the station's songs out of new batches once reached; it
is not a provider-enforced spending cap and does not account for other Runpod workloads.
Runpod itself offers no monthly budget, only the prepaid balance and an hourly
account spend limit; when the balance is used up, Runpod stops the worker.
Existing workers, billing delay and ancillary charges can exceed the estimate.
Manage account funding and billing separately in the Runpod console. Deleting a
stopped worker keeps its recorded end time. There is no daily-budget setting.

Other `[runpod]` settings control allowed GPU types and relative speed, minimum
speed, worker disk size, wait time, maximum batch runtime and parallel lanes.
Defaults use RTX 4090, an 80 GB worker disk, three lanes, 7 GB reserved VRAM per lane,
ten-song chunks and staggered lane startup. An idle worker retires after its idle timeout;
uncopied reports cause it to be stopped, not automatically deleted.

Each station sets an **idle timeout** of 1 to 60 minutes (default 5): how long the GPU worker
stays after its last batch, so that further songs start without a new worker. A shared worker
uses the longest idle timeout of its stations. Five minutes is about as long as starting a new
worker takes, so waiting longer than a new start would cost rarely pays off.

## How detection works

The first 30 seconds of each original file are analyzed. Demucs and Mel-Band separate
vocals; Whisper, Qwen with forced alignment, PANNs and SMAD provide independent timing
evidence. One per-recording ASR language is detected from its Mel-Band stem. A frozen
ExtraTrees selector fits from packaged derived features, then vocal guards and the
resolver produce the result. It does not learn continuously from new reviews.

The current resolver uses musical timing only when the opening is independently
verified vocal-free or a listener has explicitly cleared it. An empty transcript is
not proof of no vocals. A vocal conflict veto prevents publication. A verified clear
opening without a feasible musical candidate maps to a conservative 30-second value.
The current resolver does not generally replace a valid vocal cue with an earlier
musical cue just because one exists.

Published values are original-file seconds: blank stays unknown, zero means no usable
intro, and a positive number is the measured handover. Only `intro_end` is updated
through the Sweepers endpoint; other custom fields are preserved.

## Intro end values

Values are seconds from the **original file start**, not from an AutoCue cue-in:

| Value | Meaning |
| --- | --- |
| Blank | Unknown or unresolved; left untouched |
| `0` | No usable intro; no sweeper |
| Positive | Measured usable intro end |

## Manual-value protection

The detector never overwrites an `intro_end` it did not write. An operator's edit,
including clearing a previously detector-owned field, locks that recording against
further detector writes. To deliberately release it:

```bash
docker compose exec detector intro-detector unlock --station 1 --media 123
```

This removes protection; it is not a cue edit. Use it only after deciding that the
detector should manage that value again.

## Commands and reviews

```bash
docker compose exec detector intro-detector check
docker compose exec detector intro-detector status
docker compose exec detector intro-detector station show
docker compose exec detector intro-detector station set 1 enabled=true budget_usd=20
docker compose exec -T detector intro-detector key set 1 < runpod-key.txt
docker compose exec detector intro-detector backfill --station 1 --dry-run
docker compose exec detector intro-detector backfill --station 1 --limit 100
docker compose exec detector intro-detector process --station 1
docker compose exec detector intro-detector decisions --station 1
docker compose exec detector intro-detector retry --station 1
docker compose exec detector intro-detector pod status --station 1
```

Stations enabled from the command line also need AzuraCast access: run
`azuracast_cli sweepers:detector-account --sync` in AzuraCast's `web` container.
`backfill --include-set` also considers already tagged files; use deliberately. Pair it
with `process --force` only when intentionally replacing existing cues. For completed
analyses, `republish --force` uses stored evidence to replace existing values; it does not
rescan audio. `retry` requeues failures. `process` may create
a paid GPU worker. Status and decisions are service data, not proof that station playback
has been validated.

Review storage belongs to this detector, not to the Sweepers plugin. Reviews bind to
the audio hash. `instrumental` confirms the first 30 seconds contain no vocals and
can override a false vocal veto, but a musical measurement is still required.
`clear_opening` records a reviewed vocal-free opening.

```bash
docker compose exec detector intro-detector review add --kind instrumental --station 1 --media 123
docker compose exec detector intro-detector review list
```

## Backups

Keep configuration, `.env`, worker keys and the data volume backed up securely; the
volume also holds the Runpod keys. Settings backups contain secrets and must not be
committed or shared publicly. For a consistent data backup, first stop any worker and
then the service:

```bash
docker compose exec detector intro-detector pod stop --station 1
docker compose stop detector
docker compose run -T --rm --no-deps --entrypoint tar detector -czf - -C /data . > detector-data.tar.gz
docker compose up -d detector
```

Preserve the complete database directory, including SQLite WAL files. Restoring an
older backup may require reconciling already published values and worker records.
Do not delete a worker containing reports that have not been recovered.

## Troubleshooting

- **API check fails:** verify HTTPS and the AzuraCast key. Missing station permissions
  are only warnings in `check`; the plugin grants them for enabled stations.
- **Runpod key rejected:** the key must manage Pods; a read-only key is insufficient.
- **Station blocked:** the Intro Detector page names the reason: no key, budget
  reached, a failed batch waiting for its retry, or a stopped worker that needs a
  decision.
- **No scans:** initial sync does not backfill. Check the pending queue, filters and
  the station's automatic processing; manual stations need "Process now".
- **No GPU capacity:** check allowed GPU types, speed and hourly price cap. Do not
  remove cost limits just to silence the error.
- **Bundled worker unavailable:** report the setup failure to the repository
  maintainers. Publishing a usable default image is their release responsibility.
  For an explicitly customized private deployment only, check its registry credentials.
- **Worker SSH failure:** confirm UID 10001 can read the mounted private key. Preserve
  the existing key while its worker is active.
- **Stopped worker with uncopied reports:** recover the reports before deleting it.
  Analysis with that key pauses without using up song attempts. Deleting the worker
  on the Intro Detector page (or `pod delete --force --station …`) discards that protection.
- **Disk full or SQLite error:** stop new processing, free disposable build/test
  artifacts and retain the data volume. Do not delete the database, WAL or analyses
  to make an error disappear. Restore from a consistent backup only after diagnosis.
- **Unexpected cue:** inspect the stored evidence and listen to the original file.
  Do not treat uncertain vocals as proof that an early musical cue is safe.

```bash
docker compose logs --tail 100 detector
docker compose exec detector intro-detector pod status
```

Remove API keys and private paths before sharing diagnostics.

## Removal

Stop the worker first, recover reports if needed, then run `docker compose down`.
This removes the service containers but preserves the data volume. Do not add
`--volumes` unless permanently deleting the stored analyses is intended and backed up.
Deleting the checkout does not erase intro values already saved in AzuraCast.

## Development and A/B comparisons

Use Linux and install the package dependencies before running:

```bash
cd tests && python3 -m unittest discover -s .
```

Setup tests use a fake Docker backend and no live API or GPU; ownership scenarios
require root inside a disposable test container. Never commit test audio, credentials,
private reviews or analysis results. Keep cache-hashed `pod/` file bytes exact unless
the corresponding inference behavior intentionally changes.

For a fixed pipeline comparison:

```bash
docker compose exec detector intro-detector ab-select --name comparison --count 20 --controls 5
docker compose exec detector intro-detector ab-run --name comparison --max-usd 1.5
docker compose exec detector intro-detector ab-report --name comparison
```

`ab-run` is paid GPU work and stores only separate B evidence. It reuses completed
entries in the set. `--max-usd` adjusts the run's estimated budget/runtime allowance
and can exceed the ordinary monthly allowance; it is not a guaranteed invoice cap.
`ab-report` evaluates A and B with the same current selector and resolver, so a
selector-version comparison needs separately frozen selector versions.

Compare against listener labels without training/holdout leakage. Earlier measured
musical handovers may be editorially acceptable, but do not label arbitrary early
cues as correct. Keep vocal accuracy and final cue suitability separate. Run
`decision-diff` before replacing stored/published decisions. Image redistribution
must also pass the requirements in [LICENSES.md](LICENSES.md).

## Maintainer release preparation

### Service-only release

For changes confined to the detector service, such as selector updates, do not rebuild
the GPU worker image. Keep the existing public `v1` worker. Create a draft release on
the exact `main` commit with a `vMAJOR.MINOR.PATCH` tag matching `pyproject.toml` and
include `GPU WORKER IMAGE: UNCHANGED` in the release notes. Publishing runs the source
release checks: tag/version and `main` ancestry validation, unit tests, and an
anonymous pull of the existing `v1` worker. The service image is rebuilt locally by
`bin/setup.sh` when the detector is updated. Run `decision-diff` before replacing
stored or published decisions.

Complete worker preparation before presenting a release as installable:

1. Build [gpu/Dockerfile](../gpu/Dockerfile) through the **GPU image** workflow.
  The image contains both environments and pinned model assets; ordinary users
  must not be asked to reproduce this build on their detector server.
2. Validate offline model loading. Changes to inference or detection also require
  the controlled comparisons described above; paid GPU work needs authorization.
3. Run **GPU image** manually from `main` with `stage_candidate=true`, `publish=false`,
  the target version, exact `main` commit, digest-pinned base image, and the future
  corresponding-source asset URL. This builds a private candidate. Confirm offline
  model loading and inspect the candidate's exact `image-legal` inventory.
4. Complete the exact-version redistribution review in [LICENSES.md](LICENSES.md) and
  against the candidate inventory. Review source obligations, LGPL replacement rights,
  NVIDIA terms and model notices before approving redistribution.
5. Create a draft GitHub Release on the `main` commit with a `vMAJOR.MINOR.PATCH` tag
  matching `pyproject.toml`. Add `REDISTRIBUTION REVIEWED: APPROVED` to its body only
  after that review, and include `GPU WORKER IMAGE: UPDATED`. Keep the release as a draft.
6. Dispatch **GPU image** again from the exact release tag ref, with the same tag,
  base-image digest and source URL. This fixes the publish build to the commit used by
  the candidate. Set `publish=true`, `generate_source_bundle=true`,
  `redistribution_reviewed=true`, `reviewed_image_id` to the candidate image ID, and
  `promote_default=true`. The workflow rebuilds and requires an exact image-ID match,
  creates and attaches corresponding source, publishes the immutable GHCR tag, checks
  an anonymous pull by tag and digest, and promotes that image to the `v1` alias.
7. Confirm release evidence records the tag, digest, source revision, source-bundle URL
  and SHA256. Only then publish the draft GitHub Release. Its release event verifies
  the existing assets, public tag/digest and `v1` alias without rebuilding the image.
8. Test installation and update in a fresh disposable environment, without GHCR
  credentials. Verify API access, container health, preservation of configuration,
  credentials and stored data, and failures before configuration changes when the
  image is unavailable. Do not start GPU processing during installation tests.

### Recovery for an existing tag

For an existing tag, never move it. Use its exact commit as `source_revision` for both
manual runs and keep all build inputs identical. Create a draft release before the
publish run so its corresponding-source URL exists. Publish the draft only after the
manual publish run has verified the image and anonymous pull.

`bin/image-context.sh` produces a deterministic tar context containing only
versioned source and license inputs. The installer and worker workflow use it;
unversioned files, credentials, audio, databases and private reports are excluded.
The derived selector training file is an explicit allowed input. The worker includes
the generic inference and evaluation helpers required by its pipeline, but the
dataset-specific listening-label injector and reference data are excluded from both
image contexts. Model coordinates and cache locks are checked against the runtime
sources. Changes to Pod source may invalidate caches.
Worker Python packages use hash-locked requirements in `gpu/`; the base image and
Debian snapshot are pinned. Deterministic context bytes do not by themselves establish
a reproducible image: clean rebuilds and the actual artifacts must also be compared
before the candidate is approved.

The detector's `health` command checks local configuration, exclusive service
ownership and a readable initialized database. It does not construct a worker,
contact APIs or migrate the database. Compose waits for this check during setup.
It is a liveness/readiness check, not proof of recent event-loop progress or remote
API availability; `check` verifies API permissions separately.

Until those gates pass, the worker release is incomplete. Keep publication and
deployment separate from source-code changes. Custom private deployments may use
`[runpod] image` and `registry_auth_id`, but those overrides must not become ordinary
operator prerequisites.
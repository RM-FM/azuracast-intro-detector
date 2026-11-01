# Advanced guide

For the ordinary install and update workflow, start with the [README](../README.md).
Commands below run from the detector checkout. Use a root shell or prefix Docker
and file-administration commands with `sudo`.

## Installation and Compose

The detector is independent of AzuraCast. It needs outbound HTTPS to AzuraCast and
Runpod and SSH/rsync access to its Runpod worker. It exposes no inbound web port.
Install Docker Engine, the Compose plugin and Git before running [setup](../bin/setup.sh).
Compose v2.20+ is needed for the setup script's readiness wait.

The checkout contains:

| Path | Purpose |
| --- | --- |
| `config.toml` | Operator settings; no API keys |
| `.env` | AzuraCast and Runpod API keys; private, never committed |
| `secrets/runpod-ssh-key` | Worker SSH private key; owned by UID 10001, mode 600 |
| `.setup-backups/` | Private copies of settings before reconfiguration |
| `detector-data` Compose volume | Database, analyses, queue, decisions and reports |

Standard installation and updates use the setup script; the manual procedures below
are alternatives for custom deployments, not tasks every operator must perform.

Setup builds the service image locally and checks the repository-provided public
worker image using registry metadata, without downloading the full image or starting
a Pod. It does not run a backfill, start processing, or change AzuraCast broadcasting. First installation
uses all visible stations, shadow mode and `auto_process = false`.

On updates, the default answer keeps all existing configuration bytes and keys.
Choosing to reconfigure changes only the host and entered keys; blank key answers
keep existing keys. Advanced settings, including the monthly budget, remain intact.
Credential checks must succeed before the detector is restarted. If setup fails,
its message gives the private technical log and settings-backup location.

For AzuraCast on the same machine, the public HTTPS URL is simplest. Private-network
HTTP is supported only when the detector joins AzuraCast's Docker network and
`[azuracast] allow_http = true` is explicitly configured. Do not enable this for
a public HTTP endpoint. Custom `COMPOSE_FILE` setups require manual installation.

## Manual installation

Clone the checkout as shown in the README, then prepare the files:

```bash
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

Set the HTTPS address and station selection in `config.toml`; set only the two API
keys in `.env`. Standard deployments use the repository's default worker image. For a
controlled first test, set `[scan] auto_process = false` and keep station mode `shadow`.
Do not regenerate an existing worker key without accounting for any active worker.

## Manual update

Allow active scans to finish, read the changes and back up configuration and data, then:

```bash
git pull --ff-only origin main
docker compose build detector
docker compose run --rm --no-deps detector check
docker compose up -d --no-deps --force-recreate --wait detector
docker compose exec -T detector intro-detector status
```

The Compose volume is retained. Do not use `down --volumes` for an update.
Automatic processing resumes according to the existing configuration after restart;
pause it beforehand if pending work should not start. Worker-image changes are
separate from service updates and do not require reanalyzing unchanged audio automatically.

## Configuration reference

[config.example.toml](../config.example.toml) is ordered from ordinary settings to
advanced calibration. Omitted fields use the service defaults. Recreate the detector
after changing configuration so file mounts also pick up atomic editor replacements:

```bash
docker compose up -d --no-deps --force-recreate --wait detector
```

### Stations and filters

`[azuracast] stations = "all"` follows the enabled stations visible to the API user,
including newly added stations. Discovery uses the authenticated dashboard endpoint,
not the public station list. A list such as `stations = [1, 2, 7]` restricts scanning.

`[station_defaults]` applies to every selected station. A `[[stations]]` block overrides
the settings for one station and always includes that station. Configurations with
only station blocks and no `stations` selection remain explicit station lists.

| Setting | Default | Meaning |
| --- | --- | --- |
| `mode` | `shadow` | `off`, propose only, or `publish` |
| `min_song_seconds` | `60` | Skip shorter files and files of unknown length |
| `exclude_playlists` | `[]` | Additional playlist IDs excluded from analysis |
| `timeout_seconds` | `60` | AzuraCast request timeout, under `[azuracast]` |

Jingle Mode playlists are always excluded. The API user needs station **View** for
discovery and **Media** for file access and intro writes. Never give the detector a
global administrator key merely to resolve missing station permissions.

### Processing

| `[scan]` setting | Service default | Meaning |
| --- | --- | --- |
| `auto_process` | `true` | Analyze newly queued files automatically; setup initially writes `false` |
| `poll_seconds` | `300` | Library discovery interval |
| `batch_quiet_seconds` | `120` | Wait after the last new upload before starting a batch |
| `batch_max_songs` | `100` | Maximum originals per batch |
| `max_attempts` | `2` | Retry limit for failures |

First discovery records the existing library without queuing it all. New or replaced
files are queued later. `backfill` queues existing unknown intros at lower priority.
`process` handles the entire pending queue; it is not a per-station processing command.

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

Start with one station. For example:

```toml
[[stations]]
id = 1
mode = "publish"
```

Keep other stations in shadow mode. Recreate and publish the checked stored decisions:

```bash
docker compose up -d --no-deps --force-recreate --wait detector
docker compose exec detector intro-detector decision-diff --station 1
docker compose exec detector intro-detector republish --station 1
```

The dry-run report lists proposed changes. `republish` uses stored analyses and does
not create a GPU worker. Check results before applying them.

Set `[scan] auto_process = true` only when new uploads may incur GPU charges
automatically. This setting is independent of station publishing mode: shadow-mode
analysis still costs money when it needs a GPU.

## Automatic GPU workers

The repository supplies the default worker-image reference and is responsible for
preparing, testing and publishing it before an operator release. Operators provide
their Runpod key; they do not build worker images, choose CUDA packages or configure
registry credentials for the standard installation.

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

The README's **US$2-$6 per 1,000 first-time tracks** is a planning range for prepared
RTX 4090 workers and batched processing, not a measured guarantee for every workload.
The GPU ranking defaults use 15 seconds per song for one lane and five minutes of
startup; three overlapping lanes can reduce wall time. Cold startup and the ten-minute
idle retirement interval matter most for small batches. Storage, retries, image
downloads, provider billing and changing rates can add cost.

Previously analyzed audio is reused by SHA256 across stations without GPU inference.
The detector host's own hosting and storage costs are separate.

Configure cost controls in the existing `[runpod]` section:

```toml
monthly_budget_usd = 50.0
max_price_per_hour = 0.90
```

The monthly limit uses estimated spend recorded by this detector for a UTC calendar
month. `0` disables it. It blocks new worker creation once reached; it is not a
provider-enforced spending cap and does not account for other Runpod workloads.
Existing workers, billing delay and ancillary charges can exceed the estimate.
Manage account funding and billing separately in the Runpod console. Setup never
asks for, resets or changes the monthly budget. There is no daily-budget setting.

Other `[runpod]` settings control allowed GPU types and relative speed, minimum
speed, worker disk size, wait time, maximum batch runtime and parallel lanes.
Defaults use RTX 4090, an 80 GB worker disk, three lanes, 7 GB reserved VRAM per lane,
ten-song chunks and staggered lane startup. An idle worker retires after 600 seconds;
uncopied reports cause it to be stopped, not automatically deleted.

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
docker compose exec detector intro-detector backfill --station 1 --dry-run
docker compose exec detector intro-detector backfill --station 1 --limit 100
docker compose exec detector intro-detector process
docker compose exec detector intro-detector decisions --station 1
docker compose exec detector intro-detector retry --station 1
docker compose exec detector intro-detector pod status
```

`backfill --include-set` also considers already tagged files; use deliberately.
`retry` requeues failures. `process` may create a paid GPU worker. Status and decisions
are service data, not proof that station playback has been validated.

Review storage belongs to this detector, not to the Sweepers plugin. Reviews bind to
the audio hash. `instrumental` confirms the first 30 seconds contain no vocals and
can override a false vocal veto, but a musical measurement is still required.
`clear_opening` records a reviewed vocal-free opening.

```bash
docker compose exec detector intro-detector review add --kind instrumental --station 1 --media 123
docker compose exec detector intro-detector review list
```

## Backups

Keep configuration, `.env`, worker keys and the data volume backed up securely.
Settings backups contain secrets and must not be committed or shared publicly.
For a consistent data backup, first stop any worker and then the service:

```bash
docker compose exec detector intro-detector pod stop
docker compose stop detector
docker compose run -T --rm --no-deps --entrypoint tar detector -czf - -C /data . > detector-data.tar.gz
docker compose up -d detector
```

Preserve the complete database directory, including SQLite WAL files. Restoring an
older backup may require reconciling already published values and worker records.
Do not delete a worker containing reports that have not been recovered.

## Troubleshooting

- **API check fails:** verify HTTPS, the keys and station View/Media permissions.
  The Runpod key must manage Pods; a read-only key is insufficient.
- **No stations:** confirm the API user's permissions and station selection. Public
  station-list visibility is not used for authenticated discovery.
- **No scans:** initial sync does not backfill. Check the pending queue, filters and
  `auto_process`; manual mode requires `process`.
- **No GPU capacity:** check allowed GPU types, speed and hourly price cap. Do not
  remove cost limits just to silence the error.
- **Bundled worker unavailable:** report the setup failure to the repository
  maintainers. Publishing a usable default image is their release responsibility.
  For an explicitly customized private deployment only, check its registry credentials.
- **Worker SSH failure:** confirm UID 10001 can read the mounted private key. Preserve
  the existing key while its worker is active.
- **Stopped worker with uncopied reports:** recover the reports before deleting it.
  A forced deletion deliberately discards that protection.
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
  after that review. Keep the release as a draft.
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
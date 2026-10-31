# AzuraCast Intro Detector

Detects song intros for the [AzuraCast Sweepers](https://github.com/RM-FM/azuracast-sweepers)
plugin. It runs on the same server or a separate Linux server and uses Runpod GPU
workers when a song needs analysis. No local GPU is required.

Starts in **shadow mode**: proposed intro values are recorded, not written to AzuraCast.
Existing manually edited intro values are protected.

Version `0.1.0` is an experimental pre-release. Review proposed cues before enabling
publishing; installation does not validate detection accuracy for your library.

## Quick Start

You need a Linux x86_64 server, Docker Engine, Docker Compose v2.20 or newer, Git
and a Runpod account.
Run the commands on the detector server, not on a GPU worker.

### Before installing

Install the Sweepers plugin on AzuraCast. Create an AzuraCast API key for a dedicated
user with **View** and **Media** permission on the stations to scan. Keep other
administrative permissions disabled.

#### Get a Runpod API key

1. Sign in to [Runpod](https://console.runpod.io/) and add funds to the account.
2. Open [Credentials](https://console.runpod.io/user/credentials), select **API Keys**
  and click **Create API Key**.
3. Name it for the detector. Give it read/write access for Pod and GPU management;
  a **Read Only** key cannot create or stop workers. Use restricted permissions
  where available. You do not need an S3 key.
4. Copy the key into a password manager. Enter it only at the setup script's hidden
  prompt; never commit it or paste it into an issue.

Worker images are prepared and maintained by this repository. Setup checks the
default image automatically; you do not need to build images or configure a registry.
If the check fails, verify network access to GHCR and retry. A missing or private
bundled image must be corrected by the maintainers, not by adding a registry login.

### Install

```bash
sudo git clone --branch main https://github.com/RM-FM/azuracast-intro-detector.git /opt/azuracast-intro-detector
cd /opt/azuracast-intro-detector
sudo bash bin/setup.sh
```

The script asks for the AzuraCast HTTPS address and both API keys. It builds the
service, verifies the worker image, prepares configuration, generates the worker SSH key,
checks API access and waits for a healthy detector. API key prompts do not display the keys.

Initial setup selects all stations visible to the API user, shadow mode and manual
processing. It does not scan the existing library or start a paid GPU worker.

Continue with [First station setup](#first-station-setup).
Other options are in [manual installation](docs/ADVANCED.md#manual-installation).

### Update

Allow any active scan to finish before updating.

```bash
cd /opt/azuracast-intro-detector
sudo git pull --ff-only origin main
sudo bash bin/setup.sh
```

Accept **Keep existing configuration** to preserve the host, keys and all settings.
Stored analyses and intro values are kept. The script rebuilds and restarts only
the detector; it does not restart AzuraCast or interrupt broadcasting.

AzuraCast's own updater does not update the detector.
For an alternative, see [manual update](docs/ADVANCED.md#manual-update).

## First station setup

1. Keep publishing disabled while preparing the station. Confirm that the desired
  stations and library filters are selected in the configuration.
2. Check the service and allow its first library sync to finish:

  ```bash
  sudo docker compose exec detector intro-detector status
  ```

3. Start with a small sample of existing songs. Replace `1` with your station ID:

  ```bash
  sudo docker compose exec detector intro-detector backfill --station 1 --limit 10
  sudo docker compose exec detector intro-detector process
  sudo docker compose exec detector intro-detector decisions --station 1
  ```

  **Processing uses paid GPU time** unless the song already has a stored analysis.
  `process` works through the whole pending queue, not only the last backfill request.

4. Listen to the sample in Sweepers' Intro Manager and compare the proposed cues
  with the songs. Shadow mode has not changed their saved intro values.
5. When satisfied, enable publishing for the intended station, restart the detector
  and republish its checked results. Enable automatic processing only when ready.
  See [publishing and automatic processing](docs/ADVANCED.md#publishing-and-automatic-processing).
6. Validate station playback with sweepers before leaving publishing enabled.

## Everyday use

New or replaced files are discovered automatically. Initial library discovery does
not queue a complete backfill. Previously analyzed audio is reused across stations.

```bash
sudo docker compose exec detector intro-detector status
sudo docker compose logs --tail 50 detector
```

### Intro end

Values are seconds from the **original file start**, not from an AutoCue cue-in:

| Value | Meaning |
| --- | --- |
| Blank | Unknown or unresolved; left untouched |
| `0` | No usable intro; no sweeper |
| Positive | Measured usable intro end |

Changes made by an operator are not overwritten. See [manual-value protection](docs/ADVANCED.md#manual-value-protection)
before deliberately unlocking a track.

### Expected cost

Allow roughly **US$2-$6 of GPU compute per 1,000 first-time tracks** with a prepared
image and batched RTX 4090 processing. This is a planning estimate, not a guaranteed
price. Small cold batches, image downloads, retries, storage and provider rates can
increase it. Reusing a stored analysis needs no GPU time.

See [costs and limits](docs/ADVANCED.md#costs-and-limits) for the assumptions and controls.

## More documentation

- [Advanced guide](docs/ADVANCED.md): optional configuration, commands,
  backups, troubleshooting, removal and development.
- [Models](docs/MODELS.md): detector models and pinned versions.
- [Third-party licenses](docs/LICENSES.md): notices and image redistribution requirements.

MIT for the detector's own code. Third-party components retain their own licenses;
see [Third-party notices](THIRD_PARTY_NOTICES.md). Independent of the AzuraCast project.

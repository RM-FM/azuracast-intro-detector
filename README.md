# AzuraCast Intro Detector

Detects song intros for the [AzuraCast Sweepers](https://github.com/RM-FM/azuracast-sweepers)
plugin. It runs next to AzuraCast and rents Runpod GPU workers when a song needs
analysis. No local GPU is required.

Each station switches it on in **Sweepers → Intro Detector** with a Runpod API key and
a monthly budget. Intros it finds are written to AzuraCast right away; intro values
entered by hand are never overwritten.

Start with one station and listen to the results; installation does not validate
detection accuracy for your library.

## Quick Start

You need AzuraCast with the Sweepers plugin on a Linux x86_64 server with Docker
Compose v2.20 or newer, Git and root access. The AzuraCast **Site Base URL**
(Administration → System Settings) must be set.

### Install

The Sweepers plugin's setup offers the detector when the plugin is installed. To add
it later, run in the plugin directory:

```bash
cd /var/azuracast/plugins/azuracast-sweepers
sudo bash bin/setup.sh --detector
```

The detector is installed to `/opt/azuracast-intro-detector`, gets its own AzuraCast
account and joins AzuraCast's Docker network. It opens no port and does not restart
AzuraCast or interrupt broadcasting. No API key is entered on the command line.

### Update

The detector is updated with the plugin: `sudo bash bin/setup.sh` in the plugin
directory moves it to the release that plugin version expects. Stored analyses,
station settings and intro values are kept.

For installation without the plugin's setup, see [manual installation](docs/ADVANCED.md#manual-installation).

## First station setup

#### Get a Runpod API key

1. Sign in to [Runpod](https://console.runpod.io/) and add funds to the account.
2. Open [Credentials](https://console.runpod.io/user/credentials), select **API Keys**
  and click **Create API Key**.
3. Name it for the detector. Give it read/write access for Pod and GPU management;
  a **Read Only** key cannot create or stop workers. You do not need an S3 key.
4. Keep the key in a password manager; it is entered only on the detector page.

Use one key for several stations or a separate key per station, whichever suits how
you pay for them. Stations with the same key share one GPU worker; each station
keeps its own monthly budget.

#### Enable the station

1. Open **Sweepers → Intro Detector** for the station. Changing it needs the station's
  broadcasting permission.
2. Switch the detector on, enter the Runpod key and a monthly budget, and save. New
  uploads are analyzed automatically from now on and their intros written to AzuraCast.
3. To analyze songs already in the library, check **Older songs** on the Status tab: it
  shows the number of songs, the estimated cost and the budget left; **Analyze them** starts.
  **Processing uses paid GPU time** unless an identical file was analyzed before.
4. Listen to the results in Sweepers' Intro Manager and correct any intro there; a
  corrected value counts as entered by hand and is kept.
5. Validate station playback with sweepers before enabling more stations.


## Everyday use

New or replaced files are discovered automatically. The detector page shows the queue,
failed songs, recent batches, the station's GPU cost this month and the Runpod balance
of its key. If several stations use the same key, it also shows how many and their
combined cost this month. An identical file that was already analyzed for any
station is reused without GPU cost.

For logs on the server:

```bash
cd /opt/azuracast-intro-detector
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

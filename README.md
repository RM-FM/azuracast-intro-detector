# AzuraCast Intro Detector

Detects song intros for the [AzuraCast Sweepers](https://github.com/RM-FM/azuracast-sweepers)
plugin. It runs next to AzuraCast and rents Runpod GPU workers when a song needs
analysis. No local GPU is required.

Each station switches it on in **Sweepers → Intro Detector** and pays with its own
Runpod account and monthly budget. Stations start in **shadow mode**: proposed intro
values are recorded, not written to AzuraCast. Intro values entered by hand are
never overwritten.

Review proposed cues before enabling publishing; installation does not validate
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

#### Enable the station

1. Open **Sweepers → Intro Detector** for the station. Changing it needs the station's
  broadcasting permission.
2. Switch the detector on, keep **Shadow** mode, enter the Runpod key and a monthly
  budget, and save. New uploads are analyzed automatically from now on.
3. To analyze songs already in the library, use **Analyze existing library**. It shows
  the number of songs and an estimated cost before anything is queued.
  **Processing uses paid GPU time** unless an identical file was analyzed before.
4. Listen to results in Sweepers' Intro Manager. Shadow mode has not changed the
  saved intro values.
5. When satisfied, switch to **Publish** mode and choose **Write them to AzuraCast** to
  publish the intros found so far.
6. Validate station playback with sweepers before leaving publishing enabled.

System administrators can instead offer the server's own Runpod account to stations,
each with a monthly limit, in the page's server-wide settings.

## Everyday use

New or replaced files are discovered automatically. The detector page shows the queue,
failed songs, recent batches, this month's GPU cost and the running worker.
Previously analyzed audio is reused across stations unless the administrator turns
sharing off.

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

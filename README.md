# AzuraCast Intro Detector

Finds where song intros end for the [AzuraCast Sweepers](https://github.com/RM-FM/azuracast-sweepers)
plugin: where the vocals start or, in songs that open without vocals, where the musical
intro ends. The result becomes the song's intro end in AzuraCast, so sweepers fit
automatically.

> [!IMPORTANT]
> Install and manage the detector through the Sweepers plugin. The plugin's setup installs
> and updates it when you choose it, and **Sweepers → Intro Detector** configures it per
> station. Manual installation is not recommended; see the [advanced guide](docs/ADVANCED.md)
> if you need it anyway.

How to switch it on, enter a Runpod key and analyze existing songs is described in the
Sweepers README under [Intro Detector](https://github.com/RM-FM/azuracast-sweepers#intro-detector-optional).

## How it works

- **Next to AzuraCast:** the detector runs in its own container on the AzuraCast server.
  It finds new uploads of the stations that use it and needs no open port.
- **GPU on demand:** songs are analyzed on a GPU rented at [Runpod](https://www.runpod.io/)
  only while there is work; no local GPU is needed. A worker starts for a batch of songs
  and shuts down after the station's idle timeout. Stations on the same Runpod account
  share one worker.
- **Analysis:** the first 30 seconds of each song are examined. The vocals are separated
  from the music, transcribed with several speech models and checked by a singing
  detector; the musical structure is measured as well. The results are combined into one
  intro end. If they do not agree, the intro end stays blank.
- **Each file once:** a file that was analyzed before, for any station, is reused at no
  cost.
- **Safe writing:** only the intro end is written. Intros entered or corrected by hand
  are never overwritten.

Analysis costs roughly USD 2 to 6 per 1,000 new songs; a library of 27,301 songs cost about
USD 3.90 per 1,000. Each station has its own monthly budget.

## More documentation

- [Advanced guide](docs/ADVANCED.md): requirements, installation and updates, manual
  installation, configuration, commands, logs, backups, troubleshooting, removal and
  development.
- [Models](docs/MODELS.md): detector models and pinned versions.
- [Third-party licenses](docs/LICENSES.md): notices and image redistribution requirements.

MIT for the detector's own code. Third-party components retain their own licenses;
see [Third-party notices](THIRD_PARTY_NOTICES.md). Independent of the AzuraCast project.
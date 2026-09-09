# O4 Dictate Local

- Deleted default dictation tool (Voxtype) to create my own local transcription module, targeting all hardware acceleration (specially NPU) and Nepali transcription as well.
- Created an Omarchy Plugin, using "/omarchy" skills and planned on 3 models, with individual runtimes.
- Designed the UI for the plugin panel with Intel branding and CPU/GPU/NPU acceleration switching.
- wired up the core daemon, still rough around the edges
- basic build/install scripts working
- added reference scripts for asr/fbank/nemo, mostly for testing
- daemon now auto starts via systemd

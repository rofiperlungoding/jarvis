# JARVIS 1.0.4 — One-click in-app auto-update

## What's new

- The "Update available" banner now offers a **Update now** button.
  Click it and JARVIS:
  1. Streams the installer from the GitHub release into
     ``%LOCALAPPDATA%\Jarvis\updates\``, with a live progress
     readout in the banner.
  2. Launches the downloaded installer silently
     (`/VERYSILENT /SUPPRESSMSGBOXES /CLOSEAPPLICATIONS
     /RESTARTAPPLICATIONS /NORESTART`).
  3. Exits so the installer can replace the binaries in place.
  4. The installer's postinstall step relaunches JARVIS on the new
     version. To the user it looks like the window vanishes for a
     moment and comes back upgraded.

- A **Release page** button stays on the banner as a fallback for
  users who want to read release notes before upgrading.

- Installer logs go to
  ``%LOCALAPPDATA%\Jarvis\updates\install-<version>.log`` for
  post-mortem when an upgrade misbehaves.

## Forward compatibility

This release is the first that knows how to auto-install. Versions
1.0.0–1.0.3 will still see the banner but will fall back to the
release page (since they don't ship the new code path). Once you're
on 1.0.4, future upgrades become one-click.

## Upgrade

Older installs will see the banner with **Open release page**.
Click it, download `JARVIS-Setup-1.0.4.exe`, and run. After that,
future versions upgrade in-place.

[JARVIS-Setup-1.0.4.exe](https://github.com/rofiperlungoding/jarvis/releases/download/v1.0.4/JARVIS-Setup-1.0.4.exe)

#!/usr/bin/env bash
set -euo pipefail

# Installs base OS packages required before running bootstrap_hunyuan.sh.

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found; this script targets Debian/Ubuntu hosts." >&2
  exit 1
fi

SUDO=""
if [[ $EUID -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "Run as root or install sudo." >&2
    exit 1
  fi
fi

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get -qq update >/dev/null
$SUDO apt-get -qq install -y \
  python3 python3-venv python3-pip \
  git curl wget ffmpeg build-essential >/dev/null

echo "Prereqs installed: python3/venv/pip, git, curl/wget, ffmpeg, build-essential."

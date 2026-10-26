"""Fail fast unless OPENROUTER_API_KEY is a regular key that can call models.

A management key passes OpenRouter's key-info route but every model call gets
401 "User not found.", so a pilot run with one burns time and writes a report
of zeros. Never prints the key.
"""

import os
import sys

import httpx

key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    sys.exit("[key] OPENROUTER_API_KEY is not set.")
try:
    resp = httpx.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": f"Bearer {key}"}, timeout=30)
except httpx.HTTPError as e:
    sys.exit(f"[key] could not reach OpenRouter: {e}")
if resp.status_code != 200:
    sys.exit(f"[key] OpenRouter rejected the key (HTTP {resp.status_code}: {resp.text[:120]}).")
data = resp.json().get("data", {})
if data.get("is_management_key") or data.get("is_provisioning_key"):
    sys.exit("[key] This is a management key; it cannot call models. "
             "Create a regular key at https://openrouter.ai/settings/keys.")
limit = data.get("limit")
print(f"[key] ok: regular key, ${data.get('usage') or 0:.2f} used, "
      f"limit {'none' if limit is None else f'${limit:.2f}'}")

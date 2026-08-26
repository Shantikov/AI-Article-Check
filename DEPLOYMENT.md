# Public backend deployment

AI Article Check needs an HTTPS backend before it can be published in the
Chrome Web Store. Ordinary users install only the extension; they do not install
Python, download the model, or run `start.cmd`.

## 1. Deploy the backend

Use `backend/Dockerfile` on a container host. The image starts with
`python -m app.run`, accepts a platform-provided `PORT`, downloads the pinned
ONNX model during startup, and does not accept analysis requests until the model
is ready. A persistent volume for the model cache avoids downloading the model
after every deployment.

Set these environment variables:

```text
APP_ENVIRONMENT=production
APP_HOST=0.0.0.0
LOCAL_MODEL_CACHE_DIR=/models
PRELOAD_MODEL=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
FETCH_TIMEOUT_SECONDS=8
FETCH_MAX_RETRIES=0
FETCH_CONCURRENCY=6
INFERENCE_CONCURRENCY=2
ALLOWED_EXTENSION_IDS=
```

The hosting service must terminate HTTPS. Do not expose the container directly
over plain HTTP. Keep `TRUST_PROXY_HEADERS=false` unless the service documents
that it removes client-supplied forwarding headers before adding its own. The
built-in rate limiter is per running container; configure an additional
platform-level quota before scaling to multiple instances.

Open `https://YOUR_API/health`. A production response contains only the status
and API version. Keep the service private or in a staging environment until the
store extension ID is known.

## 2. Build the store extension

From the project root:

```powershell
python scripts\build_release.py --public-api-base https://YOUR_API
```

This creates `dist/AI-Article-Check-VERSION-store.zip`. Its manifest contains
only the deployed API origin, and its API address is fixed at build time. The
local `127.0.0.1` build is generated separately and remains unchanged.

Upload the store ZIP as a draft in the Chrome Web Store developer dashboard.
After Google assigns the extension ID, set it on the backend:

```text
ALLOWED_EXTENSION_IDS=abcdefghijklmnopqrstuvwxyzabcdef
```

For an unpacked staging build, add its ID to the same comma-separated value.
Restart/redeploy the backend, then test URL analysis, manual page analysis,
rate-limit errors, and all Google result controls.

## 3. Release checklist

- verify the public privacy page at `https://YOUR_API/privacy`;
- complete the listing text and permission declarations from `STORE_LISTING.md`;
- add at least one real 1280x800 screenshot; the owned icons and promotional
  tile are already included under `store-assets`;
- verify the exact production extension ID in `ALLOWED_EXTENSION_IDS`;
- verify that no local API address appears in the store ZIP;
- configure hosting logs and retention to match the published privacy policy;
- set platform budgets, request quotas, health monitoring, and abuse protection;
- submit the store draft for review.

Do not publish a store ZIP built with an example or temporary API address.

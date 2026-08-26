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
INFERENCE_BATCH_SIZE=14
INFERENCE_BATCH_WAIT_MS=40
TRUST_PROXY_HEADERS=true
```

The hosting service must terminate HTTPS. Do not expose the container directly
over plain HTTP. Railway supplies a validated `X-Real-IP` header, so its
deployment should use `TRUST_PROXY_HEADERS=true`. Keep the setting false on
other hosts unless they document that their proxy replaces client-supplied
forwarding headers. The built-in rate limiter is per running container;
configure an additional platform-level quota before scaling to multiple
instances.

`INFERENCE_BATCH_SIZE=14` is the safe default for the 1 GB Railway trial
container. It does not reduce the six checked sites or the seven samples kept
per article; it only splits a larger queue into bounded ONNX runs. After moving
to a larger Hobby container, set `INFERENCE_BATCH_SIZE=42` and redeploy. No
extension update is required. Performance log lines contain only stage timings,
batch sizes, and status values; they do not contain URLs or article text.

Open `https://YOUR_API/health`. A production response contains only the status
and API version.

## 2. Build the store extension

From the project root:

```powershell
python scripts\build_release.py --public-api-base https://YOUR_API
```

This creates `dist/AI-Article-Check-VERSION-store.zip`. Its manifest contains
only the deployed API origin, and its API address is fixed at build time. The
local `127.0.0.1` build is generated separately and remains unchanged.

Upload the store ZIP as a draft in the Chrome Web Store developer dashboard.
Chrome extension service workers use the exact API origin declared in
`host_permissions`, so the permanent extension ID is not required for normal
API requests. `ALLOWED_EXTENSION_IDS` is an optional CORS response setting for
browser-page integrations; it is not authentication or abuse protection. Test
URL analysis, manual page analysis, rate-limit errors, and all Google result
controls with the final ZIP before submission.

## 3. Release checklist

- verify the public privacy page at `https://YOUR_API/privacy`;
- complete the listing text and permission declarations from `STORE_LISTING.md`;
- add at least one real 1280x800 screenshot; the owned icons and promotional
  tile are already included under `store-assets`;
- verify that no local API address appears in the store ZIP;
- configure hosting logs and retention to match the published privacy policy;
- set platform budgets, request quotas, health monitoring, and abuse protection;
- submit the store draft for review.

Do not publish a store ZIP built with an example or temporary API address.

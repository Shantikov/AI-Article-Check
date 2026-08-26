import httpx


class ExternalDetector:
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def analyze(self, text: str) -> tuple[float, float | None]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.url,
                json={"text": text},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()

        probability = float(payload["ai_probability"])
        confidence_value = payload.get("confidence")
        confidence = float(confidence_value) if confidence_value is not None else None
        if not 0 <= probability <= 1:
            raise ValueError("ai_probability must be between 0 and 1")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return probability, confidence


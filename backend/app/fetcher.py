import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx


class FetchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "network_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class FetchedPage:
    final_url: str
    html: str
    truncated: bool = False


@dataclass(frozen=True)
class ResolvedPublicUrl:
    url: str
    addresses: tuple[str, ...]


async def read_limited_body(
    response: httpx.Response,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Read up to max_bytes and report whether the response was truncated."""
    body = bytearray()
    truncated = False

    content_length = response.headers.get("content-length")
    if content_length:
        try:
            truncated = int(content_length) > max_bytes
        except ValueError:
            pass

    async for chunk in response.aiter_bytes():
        remaining = max_bytes - len(body)
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            body.extend(chunk[:remaining])
            truncated = True
            break
        body.extend(chunk)

    return bytes(body), truncated


def canonicalize_url(raw_url: str) -> str:
    try:
        parts = urlsplit(raw_url.strip())
    except ValueError as exc:
        raise FetchError("Invalid URL", code="invalid_url") from exc

    if parts.scheme.lower() not in {"http", "https"}:
        raise FetchError(
            "Only HTTP and HTTPS URLs are supported",
            code="invalid_url",
        )
    if not parts.hostname or parts.username or parts.password:
        raise FetchError("Invalid website address", code="invalid_url")

    try:
        host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise FetchError("Invalid website address", code="invalid_url") from exc
    try:
        port = parts.port
    except ValueError as exc:
        raise FetchError("Invalid port", code="invalid_url") from exc
    expected_port = 80 if parts.scheme.lower() == "http" else 443
    if port is not None and port != expected_port:
        raise FetchError(
            "Only standard web ports are supported",
            code="invalid_url",
        )

    default_port = (parts.scheme.lower() == "http" and port == 80) or (
        parts.scheme.lower() == "https" and port == 443
    )
    display_host = f"[{host}]" if ":" in host else host
    netloc = display_host if not port or default_port else f"{display_host}:{port}"
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))


def _is_forbidden_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    # `is_private` does not cover every non-public range. In particular, RFC
    # 6598 shared address space (100.64.0.0/10) is neither private nor globally
    # reachable. A public URL fetcher must accept only globally routable IPs.
    return not ip.is_global


async def resolve_public_url(raw_url: str) -> ResolvedPublicUrl:
    url = canonicalize_url(raw_url)
    parts = urlsplit(url)
    if not parts.hostname:
        raise FetchError("Invalid website address", code="invalid_url")

    if parts.hostname in {"localhost", "localhost.localdomain"}:
        raise FetchError(
            "Local network addresses are not allowed",
            code="private_address",
        )

    try:
        literal = ipaddress.ip_address(parts.hostname)
    except ValueError:
        literal = None
    if literal and _is_forbidden_ip(str(literal)):
        raise FetchError(
            "Private network addresses are not allowed",
            code="private_address",
        )

    port = parts.port or (443 if parts.scheme == "https" else 80)
    loop = asyncio.get_running_loop()
    try:
        addresses = await loop.getaddrinfo(
            parts.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise FetchError("Domain not found", code="dns_error", retryable=True) from exc

    if not addresses:
        raise FetchError("Domain not found", code="dns_error", retryable=True)
    public_addresses: list[str] = []
    for address in addresses:
        candidate = str(ipaddress.ip_address(address[4][0]))
        if _is_forbidden_ip(candidate):
            raise FetchError(
                "The domain resolves to a private network",
                code="private_address",
            )
        if candidate not in public_addresses:
            public_addresses.append(candidate)
    return ResolvedPublicUrl(url=url, addresses=tuple(public_addresses))


async def validate_public_url(raw_url: str) -> str:
    return (await resolve_public_url(raw_url)).url


def pinned_request_target(url: str, address: str) -> tuple[str, str, dict[str, str]]:
    parts = urlsplit(url)
    if not parts.hostname:
        raise FetchError("Invalid website address", code="invalid_url")
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    default_port = (parts.scheme == "http" and port == 80) or (
        parts.scheme == "https" and port == 443
    )
    display_address = f"[{address}]" if ":" in address else address
    transport_netloc = (
        display_address if default_port else f"{display_address}:{port}"
    )
    display_host = f"[{host}]" if ":" in host else host
    host_header = display_host if default_port else f"{display_host}:{port}"
    transport_url = urlunsplit(
        (parts.scheme, transport_netloc, parts.path, parts.query, "")
    )
    extensions = {"sni_hostname": host} if parts.scheme == "https" else {}
    return transport_url, host_header, extensions


def error_for_http_status(status_code: int) -> FetchError:
    if status_code in {401, 403}:
        return FetchError(
            f"The website returned HTTP {status_code} to the backend",
            code="http_access_blocked",
        )
    if status_code in {404, 410}:
        return FetchError("The page was not found", code="page_not_found")
    if status_code == 429:
        return FetchError(
            "The website is temporarily rate-limiting requests",
            code="rate_limited",
            retryable=True,
        )
    if status_code in {408, 425}:
        return FetchError(
            "The website did not complete the request",
            code="timeout",
            retryable=True,
        )
    if 500 <= status_code <= 599:
        return FetchError(
            f"The website returned a temporary server error ({status_code})",
            code="server_error",
            retryable=True,
        )
    return FetchError(
        f"The website returned HTTP {status_code}",
        code="http_error",
    )


async def _fetch_once(
    initial_url: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_redirects: int,
) -> FetchedPage:
    current_url = initial_url
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AIArticleCheck/0.9.7; "
            "+https://localhost.invalid)"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        "Accept-Language": "en,ru;q=0.8,*;q=0.5",
    }

    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        limits=limits,
        trust_env=False,
    ) as client:
        for redirect_count in range(max_redirects + 1):
            try:
                async with asyncio.timeout(timeout_seconds):
                    resolved = await resolve_public_url(current_url)
                    last_transport_error: httpx.TransportError | None = None
                    redirect_location: str | None = None
                    for address in resolved.addresses:
                        transport_url, host_header, extensions = pinned_request_target(
                            resolved.url,
                            address,
                        )
                        try:
                            async with client.stream(
                                "GET",
                                transport_url,
                                headers={"Host": host_header},
                                extensions=extensions,
                                follow_redirects=False,
                            ) as response:
                                if response.status_code in {301, 302, 303, 307, 308}:
                                    redirect_location = response.headers.get("location")
                                    if not redirect_location:
                                        raise FetchError(
                                            "Redirect response has no destination",
                                            code="invalid_response",
                                        )
                                    break

                                if response.status_code >= 400:
                                    raise error_for_http_status(response.status_code)

                                content_type = response.headers.get(
                                    "content-type", ""
                                ).lower()
                                if content_type and not any(
                                    value in content_type
                                    for value in ("text/html", "application/xhtml+xml")
                                ):
                                    raise FetchError(
                                        "The URL does not return an HTML page",
                                        code="non_html",
                                    )

                                body, truncated = await read_limited_body(
                                    response,
                                    max_bytes,
                                )
                                encoding = response.encoding or "utf-8"
                                html = body.decode(encoding, errors="replace")
                                return FetchedPage(
                                    final_url=resolved.url,
                                    html=html,
                                    truncated=truncated,
                                )
                        except httpx.TransportError as exc:
                            last_transport_error = exc
                            continue

                    if redirect_location is not None:
                        if redirect_count >= max_redirects:
                            raise FetchError(
                                "Too many redirects",
                                code="too_many_redirects",
                            )
                        current_url = canonicalize_url(
                            urljoin(resolved.url, redirect_location)
                        )
                        continue
                    if last_transport_error is not None:
                        raise last_transport_error
                    raise FetchError(
                        "Domain not found",
                        code="dns_error",
                        retryable=True,
                    )
            except FetchError:
                raise
            except (TimeoutError, httpx.TimeoutException) as exc:
                raise FetchError(
                    "The website did not respond in time",
                    code="timeout",
                    retryable=True,
                ) from exc
            except httpx.TransportError as exc:
                raise FetchError(
                    "Could not connect to the website",
                    code="network_error",
                    retryable=True,
                ) from exc
            except ValueError as exc:
                raise FetchError(
                    "Invalid response from the website",
                    code="invalid_response",
                ) from exc

    raise FetchError("Could not download the page", retryable=True)


async def fetch_html(
    raw_url: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_redirects: int = 5,
    max_retries: int = 1,
) -> FetchedPage:
    initial_url = canonicalize_url(raw_url)
    last_error: FetchError | None = None
    for attempt in range(max(0, max_retries) + 1):
        try:
            return await _fetch_once(
                initial_url,
                timeout_seconds=timeout_seconds,
                max_bytes=max_bytes,
                max_redirects=max_redirects,
            )
        except FetchError as exc:
            last_error = exc
            if not exc.retryable or attempt >= max_retries:
                raise
            await asyncio.sleep(0.35 * (attempt + 1))

    if last_error is None:
        raise FetchError("Could not download the page", retryable=True)
    raise last_error

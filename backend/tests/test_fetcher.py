import httpx
import pytest

from app.fetcher import (
    FetchError,
    canonicalize_url,
    error_for_http_status,
    pinned_request_target,
    read_limited_body,
    validate_public_url,
)


def test_canonicalize_removes_fragment_and_default_port() -> None:
    assert (
        canonicalize_url("https://Example.COM:443/story?q=1#section")
        == "https://example.com/story?q=1"
    )


def test_canonicalize_formats_ipv6_literal_safely() -> None:
    assert canonicalize_url("https://[2606:4700:4700::1111]:443/story") == (
        "https://[2606:4700:4700::1111]/story"
    )


def test_canonicalize_rejects_non_web_ports() -> None:
    with pytest.raises(FetchError, match="standard web ports"):
        canonicalize_url("https://example.com:8443/story")


def test_pinned_https_target_preserves_host_and_sni() -> None:
    target, host_header, extensions = pinned_request_target(
        "https://example.com/story?q=1",
        "203.0.113.10",
    )

    assert target == "https://203.0.113.10/story?q=1"
    assert host_header == "example.com"
    assert extensions == {"sni_hostname": "example.com"}


@pytest.mark.asyncio
async def test_rejects_loopback_address() -> None:
    with pytest.raises(FetchError):
        await validate_public_url("http://127.0.0.1/private")


@pytest.mark.asyncio
async def test_rejects_localhost() -> None:
    with pytest.raises(FetchError):
        await validate_public_url("http://localhost/private")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://100.64.0.1/private",
        "http://100.127.255.254/private",
        "http://[fc00::1]/private",
        "http://[2001:db8::1]/private",
    ],
)
async def test_rejects_every_non_global_literal_address(url: str) -> None:
    with pytest.raises(FetchError) as error:
        await validate_public_url(url)

    assert error.value.code == "private_address"


@pytest.mark.asyncio
async def test_large_response_is_truncated_instead_of_rejected() -> None:
    response = httpx.Response(
        200,
        content=b"x" * 200,
        headers={"content-length": "200"},
    )
    body, truncated = await read_limited_body(response, max_bytes=80)
    assert body == b"x" * 80
    assert truncated is True


def test_http_errors_have_specific_codes_and_retry_rules() -> None:
    blocked = error_for_http_status(403)
    missing = error_for_http_status(404)
    limited = error_for_http_status(429)
    server = error_for_http_status(503)

    assert blocked.code == "http_access_blocked"
    assert blocked.retryable is False
    assert missing.code == "page_not_found"
    assert missing.retryable is False
    assert limited.code == "rate_limited"
    assert limited.retryable is True
    assert server.code == "server_error"
    assert server.retryable is True

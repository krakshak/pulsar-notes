"""Automated Test Suite for Phase 6: 100% Local Frontend UI (Zero CDN & Strict CSP).

Tests:
1. test_zero_external_urls_in_frontend: Scans all frontend files with regex to verify zero external
   CDN, font, or analytics URLs exist.
2. test_index_injects_session_token_and_csp: Verifies GET / injects the per-boot SESSION_TOKEN,
   includes strict CSP meta & HTTP headers, and serves all local static assets cleanly.
"""

from pathlib import Path
import re
from fastapi.testclient import TestClient

from backend import config, database, main

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def test_zero_external_urls_in_frontend():
    """Scans frontend HTML, CSS, JS, and vendor assets to assert zero external URLs exist."""
    assert FRONTEND_DIR.exists(), "frontend/ directory does not exist"

    target_files = [
        FRONTEND_DIR / "index.html",
        FRONTEND_DIR / "styles.css",
        FRONTEND_DIR / "app.js",
        FRONTEND_DIR / "vendor" / "base.css",
        FRONTEND_DIR / "vendor" / "icons.js",
    ]

    # Match https://, http://, or protocol-relative //urls (excluding CSS comments /* ... */)
    url_pattern = re.compile(r"(https?://[^\s\"'<>)]+|(?<![a-zA-Z0-9:])//[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")

    allowed_literals = {
        "http://127.0.0.1:8000",
        "http://www.w3.org/2000/svg",  # Standard W3C SVG DOM namespace (not a network request)
    }

    for file_path in target_files:
        assert file_path.exists(), f"Missing expected frontend file: {file_path}"
        content = file_path.read_text(encoding="utf-8")
        # Strip JS/CSS block comments before checking for protocol-relative //
        stripped = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        matches = url_pattern.findall(stripped)
        external_matches = [m for m in matches if m not in allowed_literals]
        assert len(external_matches) == 0, (
            f"Found external URL(s) in {file_path.name}: {external_matches}"
        )


def test_index_injects_session_token_and_csp(tmp_path: Path, monkeypatch):
    """Requests GET / and verifies strict CSP meta tag, injected SESSION_TOKEN, and static assets."""
    test_db = tmp_path / "test_frontend.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db(test_db)

    with TestClient(main.app, base_url="http://127.0.0.1:8000") as client:
        resp = client.get("/")
        assert resp.status_code == 200

        # Verify CSP HTTP header
        csp_header = resp.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp_header
        assert "connect-src 'self' http://127.0.0.1:8000 ws://127.0.0.1:8000" in csp_header

        # Verify CSP meta tag and injected per-boot SESSION_TOKEN in HTML
        html = resp.text
        assert 'http-equiv="Content-Security-Policy"' in html
        assert "__LOCAL_SESSION_TOKEN__" not in html
        assert f'content="{config.SESSION_TOKEN}"' in html

        # Verify local vendor and app assets all serve 200 OK
        for asset_url in [
            "/static/vendor/base.css",
            "/static/vendor/icons.js",
            "/static/styles.css",
            "/static/app.js",
        ]:
            asset_resp = client.get(asset_url)
            assert asset_resp.status_code == 200, f"Failed to serve {asset_url}"


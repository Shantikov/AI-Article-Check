import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_product_name_is_consistent_in_extension_ui() -> None:
    manifest = json.loads(
        (ROOT / "extension" / "manifest.json").read_text(encoding="utf-8")
    )
    html = (ROOT / "extension" / "popup.html").read_text(encoding="utf-8")

    assert manifest["name"] == "AI Article Check"
    assert manifest["version"] == "0.9.3"
    assert manifest["icons"] == {
        "16": "icons/icon16.png",
        "32": "icons/icon32.png",
        "48": "icons/icon48.png",
        "128": "icons/icon128.png",
    }
    assert manifest["action"]["default_icon"] == {
        "16": "icons/icon16.png",
        "32": "icons/icon32.png",
    }
    assert manifest["action"]["default_title"] == "AI Article Check"
    assert "<title>AI Article Check</title>" in html
    assert "<h1>AI Article Check</h1>" in html
    assert "AI Content Signal" not in html
    assert 'src="icons/icon32.png"' in html
    assert "optional_host_permissions" not in manifest


def test_popup_contains_only_user_facing_controls() -> None:
    html = (ROOT / "extension" / "popup.html").read_text(encoding="utf-8")

    assert "Analyze this page" in html
    assert "Auto-check on Google" in html
    assert "Advanced settings" not in html
    assert "Server address" not in html
    assert "Calibration" not in html
    assert "Model" not in html
    assert "server-status" not in html
    assert "status-message" not in html
    assert "Automatically check" not in html
    assert ">Save<" not in html
    assert 'id="analyze-current-page" class="primary-button" type="button" disabled' in html
    assert 'id="analysis-details-toggle"' in html
    assert "View details" in html


def test_auto_check_setting_is_saved_without_main_form_submission() -> None:
    script = (ROOT / "extension" / "popup.js").read_text(encoding="utf-8")

    assert 'autoResultsSelect.addEventListener("change"' in script
    assert 'autoResults: Number(autoResultsSelect.value)' in script
    assert "refreshStatus" not in script
    assert "PING_BACKEND" not in script
    assert "Server offline" not in script
    assert 'summary: "Service temporarily unavailable"' in script
    assert 'type: "GET_CURRENT_TAB_CONTEXT"' in script
    assert "setCurrentPageAvailability" in script
    assert "prepareAnalysisDetails" in script
    assert 'appendEvidenceSection("Evidence for AI"' in script
    assert 'appendEvidenceSection("Evidence against AI"' in script
    assert "reason.detail || detailedEvidenceMessage(reason.message)" in script
    assert 'excerpt.className = "evidence-excerpt"' in script


def test_google_results_are_rendered_as_each_check_finishes() -> None:
    content_script = (ROOT / "extension" / "content.js").read_text(encoding="utf-8")
    background_script = (ROOT / "extension" / "background.js").read_text(
        encoding="utf-8"
    )

    assert "await Promise.all(pending.map(analyzeSearchTarget));" in content_script
    assert 'urls: [target.url]' in content_script
    assert "let cacheMutationQueue = Promise.resolve();" in background_script
    assert "function mutateCache(mutator)" in background_script
    assert 'excerpt.className = "acs-details-excerpt"' in content_script

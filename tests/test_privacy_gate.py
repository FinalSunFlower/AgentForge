from pathlib import Path

from apps.mobile_contract.privacy_gate import scan


def test_privacy_gate_detects_sensitive_fields(tmp_path: Path) -> None:
    source = tmp_path / "screen.tsx"
    source.write_text("const password = form.password;", encoding="utf-8")
    assert scan(tmp_path)
    assert not scan(tmp_path, {"password"})


def test_privacy_gate_ignores_dependencies(tmp_path: Path) -> None:
    dependency = tmp_path / "node_modules" / "package.js"
    dependency.parent.mkdir()
    dependency.write_text("const camera = true", encoding="utf-8")
    assert scan(tmp_path) == []

import json

from local_vision_mcp.cli import settings


def test_installed_settings_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    home = tmp_path / "installation"
    home.mkdir()
    media = tmp_path / "media"
    media.mkdir()
    (home / "settings.json").write_text(
        json.dumps(
            {
                "roots": [str(media)],
                "config": {"vision_model": "installed-model", "omni_enabled": False},
            }
        )
    )
    (tmp_path / ".local-vision.json").write_text(json.dumps({"vision_model": "checkout-model"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCAL_VISION_CONFIG", raising=False)
    monkeypatch.delenv("LOCAL_VISION_ROOTS", raising=False)
    config, roots = settings(home)
    assert config["vision_model"] == "installed-model"
    assert roots == (media,)
    monkeypatch.setenv("LOCAL_VISION_CONFIG", str(tmp_path / ".local-vision.json"))
    assert settings(home)[0]["vision_model"] == "checkout-model"

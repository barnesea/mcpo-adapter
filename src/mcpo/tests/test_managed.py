from unittest.mock import patch

from mcpo.utils.managed import (
    InstallManager,
    RuntimeManager,
    normalize_managed_server_spec,
    normalize_served_protocol,
    validate_server_config_extensions,
)


def test_normalize_protocol_alias():
    assert normalize_served_protocol("http-streamable") == "streamable-http"
    assert normalize_served_protocol("streamable_http") == "streamable-http"


def test_validate_extensions_accepts_managed_runtime_upstream():
    cfg = {
        "upstream": {"type": "stdio", "command": "npx", "args": ["-y", "@scope/server"]},
        "runtime": {"mode": "host"},
        "serve_protocols": ["sse", "http-streamable"],
    }
    validate_server_config_extensions("playwright", cfg)


def test_normalize_managed_server_spec_legacy_stdio():
    cfg = {"command": "uvx", "args": ["mcp-server-time"]}
    spec = normalize_managed_server_spec("time", cfg)
    assert spec.upstream.server_type == "stdio"
    assert spec.upstream.command == "uvx"
    assert spec.upstream.args == ["mcp-server-time"]


def test_normalize_managed_server_spec_uses_runtime_command_when_upstream_missing():
    cfg = {
        "runtime": {"mode": "host", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
        "serve_protocols": ["sse"],
    }
    spec = normalize_managed_server_spec("memory", cfg)
    assert spec.upstream.server_type == "stdio"
    assert spec.upstream.command == "npx"
    assert spec.upstream.args == ["-y", "@modelcontextprotocol/server-memory"]


def test_install_manager_skips_when_marker_exists(tmp_path):
    cache_dir = tmp_path / "cache"
    marker = cache_dir / "install.marker.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}")

    spec = normalize_managed_server_spec(
        "memory",
        {
            "install": {
                "backend": "npx",
                "package": "@modelcontextprotocol/server-memory",
                "cache_dir": str(cache_dir),
            },
            "upstream": {"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
        },
    )
    manager = InstallManager(spec)
    with patch("subprocess.run") as mock_run:
        manager.prepare()
    mock_run.assert_not_called()


def test_install_manager_runs_and_writes_marker(tmp_path):
    cache_dir = tmp_path / "cache"
    spec = normalize_managed_server_spec(
        "memory",
        {
            "install": {
                "backend": "npx",
                "package": "@modelcontextprotocol/server-memory",
                "cache_dir": str(cache_dir),
            },
            "upstream": {"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
        },
    )
    manager = InstallManager(spec)
    with patch("subprocess.run") as mock_run:
        manager.prepare()
    mock_run.assert_called_once()
    assert (cache_dir / "install.marker.json").exists()


def test_runtime_manager_resolves_docker_stdio_command():
    spec = normalize_managed_server_spec(
        "memory",
        {
            "runtime": {
                "mode": "docker",
                "image": "ghcr.io/open-webui/mcpo:main",
                "bind_mounts": ["/tmp/a:/data/a"],
            },
            "upstream": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
            },
        },
    )
    resolved = RuntimeManager(spec).resolve_upstream()
    assert resolved.server_type == "stdio"
    assert resolved.command == "docker"
    assert "run" in resolved.args
    assert "ghcr.io/open-webui/mcpo:main" in resolved.args

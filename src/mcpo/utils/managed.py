import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcpo.utils.main import normalize_server_type

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_IMAGE = os.getenv("MCPO_MANAGED_RUNTIME_IMAGE", "ghcr.io/open-webui/mcpo:main")
DEFAULT_CACHE_ROOT = Path(os.path.expanduser("~/.mcpo/cache"))
SUPPORTED_INSTALLERS = {"npx", "npm", "uvx", "command"}
SUPPORTED_RUNTIME_MODES = {"host", "docker"}
SUPPORTED_PROTOCOLS = {"sse", "streamable-http"}


def normalize_served_protocol(protocol: str) -> str:
    if not isinstance(protocol, str):
        raise ValueError("Protocol values must be strings")
    value = protocol.strip().lower()
    if value == "http-streamable":
        return "streamable-http"
    return normalize_server_type(value)


@dataclass
class InstallSpec:
    backend: str
    package: Optional[str] = None
    install_args: List[str] = field(default_factory=list)
    exec_args: List[str] = field(default_factory=list)
    command: Optional[List[str]] = None
    cache_dir: Optional[str] = None
    marker_file: Optional[str] = None
    auto_install: bool = True


@dataclass
class RuntimeSpec:
    mode: str = "host"
    image: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    ports: List[str] = field(default_factory=list)
    bind_mounts: List[str] = field(default_factory=list)
    volumes: List[str] = field(default_factory=list)
    startup_timeout_seconds: float = 20.0


@dataclass
class UpstreamSpec:
    server_type: str
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class ManagedServerSpec:
    server_name: str
    install: Optional[InstallSpec]
    runtime: RuntimeSpec
    upstream: UpstreamSpec
    serve_protocols: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)
    oauth_config: Optional[Dict[str, Any]] = None
    client_header_forwarding: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_managed(self) -> bool:
        return self.install is not None or "runtime" in self.raw_config or "upstream" in self.raw_config


@dataclass
class RuntimeHandle:
    process: Optional[subprocess.Popen] = None
    container_id: Optional[str] = None
    cleanup_command: Optional[List[str]] = None


def _ensure_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"'{field_name}' must contain only strings")
    return value


def _ensure_dict_str_str(value: Any, field_name: str) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be an object")
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"'{field_name}' must contain only string:string pairs")
    return value


def validate_server_config_extensions(server_name: str, server_cfg: Dict[str, Any]) -> None:
    install_cfg = server_cfg.get("install")
    if install_cfg is not None:
        if not isinstance(install_cfg, dict):
            raise ValueError(f"Server '{server_name}' 'install' must be an object")
        backend = install_cfg.get("backend", "command")
        if backend not in SUPPORTED_INSTALLERS:
            raise ValueError(f"Server '{server_name}' install backend '{backend}' is not supported")
        _ensure_list(install_cfg.get("install_args"), "install.install_args")
        _ensure_list(install_cfg.get("exec_args"), "install.exec_args")
        if install_cfg.get("command") is not None:
            _ensure_list(install_cfg.get("command"), "install.command")

    runtime_cfg = server_cfg.get("runtime")
    if runtime_cfg is not None:
        if not isinstance(runtime_cfg, dict):
            raise ValueError(f"Server '{server_name}' 'runtime' must be an object")
        mode = runtime_cfg.get("mode", "host")
        if mode not in SUPPORTED_RUNTIME_MODES:
            raise ValueError(f"Server '{server_name}' runtime mode '{mode}' is not supported")
        _ensure_list(runtime_cfg.get("args"), "runtime.args")
        _ensure_list(runtime_cfg.get("ports"), "runtime.ports")
        _ensure_list(runtime_cfg.get("bind_mounts"), "runtime.bind_mounts")
        _ensure_list(runtime_cfg.get("volumes"), "runtime.volumes")
        _ensure_dict_str_str(runtime_cfg.get("env"), "runtime.env")

    upstream_cfg = server_cfg.get("upstream")
    if upstream_cfg is not None:
        if not isinstance(upstream_cfg, dict):
            raise ValueError(f"Server '{server_name}' 'upstream' must be an object")
        upstream_type = normalize_server_type(upstream_cfg.get("type"))
        if upstream_type not in ("stdio", "sse", "streamable-http"):
            raise ValueError(
                f"Server '{server_name}' upstream type must be one of stdio, sse, streamable-http"
            )
        if upstream_type == "stdio":
            if not upstream_cfg.get("command") and not server_cfg.get("runtime", {}).get("command"):
                raise ValueError(
                    f"Server '{server_name}' stdio upstream requires 'upstream.command' or 'runtime.command'"
                )
        elif not upstream_cfg.get("url"):
            raise ValueError(f"Server '{server_name}' upstream type '{upstream_type}' requires 'url'")

    served_protocols = server_cfg.get("serve_protocols")
    if served_protocols is not None:
        protocols = _ensure_list(served_protocols, "serve_protocols")
        for protocol in protocols:
            normalized = normalize_served_protocol(protocol)
            if normalized not in SUPPORTED_PROTOCOLS:
                raise ValueError(
                    f"Server '{server_name}' protocol '{protocol}' is unsupported (expected sse or streamable-http)"
                )

    # If runtime-only managed config is used without legacy keys and without explicit upstream,
    # require a runtime command so we can infer stdio upstream.
    if (
        server_cfg.get("runtime")
        and not upstream_cfg
        and not server_cfg.get("command")
        and not server_cfg.get("url")
        and not server_cfg.get("type")
        and not server_cfg.get("runtime", {}).get("command")
    ):
        raise ValueError(
            f"Server '{server_name}' runtime-only config requires runtime.command when upstream is omitted"
        )


def normalize_managed_server_spec(server_name: str, server_cfg: Dict[str, Any]) -> ManagedServerSpec:
    runtime_cfg = server_cfg.get("runtime", {}) or {}
    upstream_cfg = server_cfg.get("upstream", {}) or {}
    install_cfg = server_cfg.get("install")

    runtime = RuntimeSpec(
        mode=runtime_cfg.get("mode", "host"),
        image=runtime_cfg.get("image"),
        command=runtime_cfg.get("command"),
        args=runtime_cfg.get("args", []) or [],
        env=runtime_cfg.get("env", {}) or {},
        workdir=runtime_cfg.get("workdir"),
        ports=runtime_cfg.get("ports", []) or [],
        bind_mounts=runtime_cfg.get("bind_mounts", []) or [],
        volumes=runtime_cfg.get("volumes", []) or [],
        startup_timeout_seconds=float(runtime_cfg.get("startup_timeout_seconds", 20.0)),
    )

    install = None
    if install_cfg:
        command = install_cfg.get("command")
        if command is not None and not isinstance(command, list):
            raise ValueError(
                f"Server '{server_name}' install.command must be an array when provided"
            )
        install = InstallSpec(
            backend=install_cfg.get("backend", "command"),
            package=install_cfg.get("package"),
            install_args=install_cfg.get("install_args", []) or [],
            exec_args=install_cfg.get("exec_args", []) or [],
            command=command,
            cache_dir=install_cfg.get("cache_dir"),
            marker_file=install_cfg.get("marker_file"),
            auto_install=install_cfg.get("auto_install", True),
        )

    legacy_server_type = normalize_server_type(server_cfg.get("type", "stdio"))
    if server_cfg.get("command"):
        legacy_server_type = "stdio"

    upstream_type = normalize_server_type(upstream_cfg.get("type", legacy_server_type))
    upstream_command = upstream_cfg.get("command", server_cfg.get("command"))
    upstream_args = upstream_cfg.get("args", server_cfg.get("args", [])) or []
    upstream_env = upstream_cfg.get("env", server_cfg.get("env", {})) or {}
    upstream_url = upstream_cfg.get("url", server_cfg.get("url"))
    upstream_headers = upstream_cfg.get("headers", server_cfg.get("headers"))

    # If managed runtime command is present and stdio upstream command is not explicit,
    # use runtime command as stdio source.
    if upstream_type == "stdio" and not upstream_command and runtime.command:
        upstream_command = runtime.command
        upstream_args = runtime.args

    # Generate stdio command defaults from install config when possible.
    if upstream_type == "stdio" and not upstream_command and install and install.package:
        if install.backend in ("npx", "npm"):
            upstream_command = "npx"
            upstream_args = ["-y", install.package] + install.exec_args
        elif install.backend == "uvx":
            upstream_command = "uvx"
            upstream_args = [install.package] + install.exec_args

    upstream = UpstreamSpec(
        server_type=upstream_type,
        command=upstream_command,
        args=upstream_args,
        env=upstream_env,
        url=upstream_url,
        headers=upstream_headers,
    )

    serve_protocols = []
    for protocol in server_cfg.get("serve_protocols", []) or []:
        normalized = normalize_served_protocol(protocol)
        if normalized not in serve_protocols:
            serve_protocols.append(normalized)

    disabled_tools = server_cfg.get("disabledTools")
    if disabled_tools is None:
        disabled_tools = server_cfg.get("disabled_tools", [])

    return ManagedServerSpec(
        server_name=server_name,
        install=install,
        runtime=runtime,
        upstream=upstream,
        serve_protocols=serve_protocols,
        disabled_tools=disabled_tools or [],
        oauth_config=server_cfg.get("oauth"),
        client_header_forwarding=server_cfg.get("client_header_forwarding", {"enabled": False}),
        raw_config=server_cfg,
    )


class InstallManager:
    def __init__(self, spec: ManagedServerSpec):
        self.spec = spec
        self.install = spec.install

    def _cache_dir(self) -> Path:
        if self.install and self.install.cache_dir:
            return Path(os.path.expanduser(self.install.cache_dir))
        return DEFAULT_CACHE_ROOT / self.spec.server_name

    def _marker_file(self) -> Path:
        if self.install and self.install.marker_file:
            return Path(os.path.expanduser(self.install.marker_file))
        return self._cache_dir() / "install.marker.json"

    def _build_install_command(self) -> Optional[List[str]]:
        if not self.install:
            return None

        if self.install.command:
            return self.install.command

        backend = self.install.backend
        package = self.install.package
        cache_dir = str(self._cache_dir())

        if backend in ("npm", "npx"):
            if not package:
                raise ValueError(f"Server '{self.spec.server_name}' install.package is required for npm/npx")
            return ["npm", "install", "--prefix", cache_dir, package] + self.install.install_args

        if backend == "uvx":
            if not package:
                raise ValueError(f"Server '{self.spec.server_name}' install.package is required for uvx")
            return ["uv", "tool", "install", package] + self.install.install_args

        return None

    def prepare(self) -> None:
        if not self.install or not self.install.auto_install:
            return

        cache_dir = self._cache_dir()
        marker_file = self._marker_file()
        cache_dir.mkdir(parents=True, exist_ok=True)

        if marker_file.exists():
            logger.info(
                "Install cache hit for managed server '%s' (%s)",
                self.spec.server_name,
                marker_file,
            )
            return

        command = self._build_install_command()
        if not command:
            return

        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(cache_dir / "uv-cache")
        env["UV_TOOL_DIR"] = str(cache_dir / "uv-tools")

        logger.info(
            "Running install for managed server '%s': %s",
            self.spec.server_name,
            shlex.join(command),
        )
        subprocess.run(command, check=True, env=env)

        marker_payload = {
            "server_name": self.spec.server_name,
            "installed_at": int(time.time()),
            "backend": self.install.backend,
            "package": self.install.package,
            "command": command,
        }
        marker_file.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")
        logger.info(
            "Install complete for managed server '%s'; marker written to %s",
            self.spec.server_name,
            marker_file,
        )


class RuntimeManager:
    def __init__(self, spec: ManagedServerSpec):
        self.spec = spec
        self.handle = RuntimeHandle()

    def _cache_dir(self) -> Path:
        if self.spec.install and self.spec.install.cache_dir:
            return Path(os.path.expanduser(self.spec.install.cache_dir))
        return DEFAULT_CACHE_ROOT / self.spec.server_name

    def _merge_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.spec.runtime.env or {})
        env.update(self.spec.upstream.env or {})
        cache_dir = self._cache_dir()
        env["UV_CACHE_DIR"] = str(cache_dir / "uv-cache")
        env["UV_TOOL_DIR"] = str(cache_dir / "uv-tools")
        if self.spec.install and self.spec.install.backend in ("npm", "npx"):
            npm_prefix = str(cache_dir)
            env["NPM_CONFIG_PREFIX"] = npm_prefix
            existing_path = env.get("PATH", "")
            node_bin = str(cache_dir / "node_modules" / ".bin")
            if node_bin not in existing_path.split(":"):
                env["PATH"] = f"{node_bin}:{existing_path}" if existing_path else node_bin
        return env

    def _resolve_stdio_command(self) -> tuple[str, List[str], Dict[str, str]]:
        command = self.spec.upstream.command
        args = self.spec.upstream.args or []
        env = self._merge_env()

        if not command:
            raise ValueError(f"Managed stdio server '{self.spec.server_name}' does not have a command")

        if self.spec.runtime.mode == "docker":
            image = self.spec.runtime.image or DEFAULT_DOCKER_IMAGE
            docker_args: List[str] = ["run", "-i", "--rm"]
            for port in self.spec.runtime.ports:
                docker_args.extend(["-p", port])
            for mount in self.spec.runtime.bind_mounts:
                docker_args.extend(["-v", mount])
            for volume in self.spec.runtime.volumes:
                docker_args.extend(["-v", volume])
            if self.spec.runtime.workdir:
                docker_args.extend(["-w", self.spec.runtime.workdir])
            for key, value in self.spec.runtime.env.items():
                docker_args.extend(["-e", f"{key}={value}"])
            docker_args.append(image)
            docker_args.append(command)
            docker_args.extend(args)
            logger.info(
                "Resolved docker stdio runtime for '%s' using image '%s'",
                self.spec.server_name,
                image,
            )
            return "docker", docker_args, os.environ.copy()

        return command, args, env

    def start_background_runtime_if_needed(self) -> None:
        # stdio mode does not need a sidecar process; stdio client owns process lifecycle.
        if self.spec.upstream.server_type == "stdio":
            return
        if not self.spec.runtime.command:
            return

        env = self._merge_env()
        command = [self.spec.runtime.command] + (self.spec.runtime.args or [])

        if self.spec.runtime.mode == "host":
            logger.info(
                "Starting managed host runtime for '%s': %s",
                self.spec.server_name,
                shlex.join(command),
            )
            self.handle.process = subprocess.Popen(
                command,
                env=env,
                cwd=self.spec.runtime.workdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        image = self.spec.runtime.image or DEFAULT_DOCKER_IMAGE
        docker_cmd: List[str] = ["docker", "run", "-d", "--rm"]
        for port in self.spec.runtime.ports:
            docker_cmd.extend(["-p", port])
        for mount in self.spec.runtime.bind_mounts:
            docker_cmd.extend(["-v", mount])
        for volume in self.spec.runtime.volumes:
            docker_cmd.extend(["-v", volume])
        if self.spec.runtime.workdir:
            docker_cmd.extend(["-w", self.spec.runtime.workdir])
        for key, value in self.spec.runtime.env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        docker_cmd.append(image)
        docker_cmd.extend(command)

        logger.info(
            "Starting managed docker runtime for '%s': %s",
            self.spec.server_name,
            shlex.join(docker_cmd),
        )
        result = subprocess.run(
            docker_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        self.handle.container_id = result.stdout.strip() or None
        if self.handle.container_id:
            self.handle.cleanup_command = ["docker", "rm", "-f", self.handle.container_id]
            logger.info(
                "Managed docker runtime started for '%s' container_id=%s",
                self.spec.server_name,
                self.handle.container_id,
            )

    def resolve_upstream(self) -> UpstreamSpec:
        upstream = self.spec.upstream
        if upstream.server_type == "stdio":
            command, args, env = self._resolve_stdio_command()
            return UpstreamSpec(
                server_type="stdio",
                command=command,
                args=args,
                env=env,
                headers=upstream.headers,
            )
        return UpstreamSpec(
            server_type=upstream.server_type,
            url=upstream.url,
            headers=upstream.headers,
            env=self._merge_env(),
        )

    def stop(self) -> None:
        if self.handle.process is not None and self.handle.process.poll() is None:
            logger.info("Stopping managed process runtime for '%s'", self.spec.server_name)
            self.handle.process.terminate()
            try:
                self.handle.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.handle.process.kill()
        if self.handle.cleanup_command:
            logger.info("Stopping managed docker runtime for '%s'", self.spec.server_name)
            subprocess.run(self.handle.cleanup_command, check=False)

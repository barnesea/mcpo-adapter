import contextlib
import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from starlette.responses import Response

from mcp import ClientSession, types
from mcp.server.lowlevel.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

logger = logging.getLogger(__name__)


def create_bridge_server(
    app: FastAPI,
    *,
    server_name: str,
) -> Server:
    bridge_server = Server(name=f"{server_name}-bridge")

    async def get_session() -> ClientSession:
        session_manager: Optional[Any] = getattr(
            app.state, "session_manager", None
        )
        if not session_manager:
            raise RuntimeError("Bridge session manager is not available")
        session, _ = await session_manager.ensure_initialized()
        app.state.session = session
        return session

    async def _handle_with_reconnect(action_name: str, invoke):
        session_manager: Optional[Any] = getattr(
            app.state, "session_manager", None
        )
        if not session_manager:
            raise RuntimeError("Bridge session manager is not available")
        try:
            session = await get_session()
            return await invoke(session)
        except Exception:
            logger.warning(
                "Bridge action '%s' failed on '%s'; attempting reconnect",
                action_name,
                server_name,
                exc_info=True,
            )
            session, _ = await session_manager.reconnect()
            app.state.session = session
            return await invoke(session)

    @bridge_server.list_tools()
    async def list_tools(req: types.ListToolsRequest) -> types.ListToolsResult:
        return await _handle_with_reconnect(
            "tools/list",
            lambda session: session.list_tools(cursor=(req.params.cursor if req.params else None)),
        )

    async def call_tool_handler(req: types.CallToolRequest) -> types.ServerResult:
        result = await _handle_with_reconnect(
            "tools/call",
            lambda session: session.call_tool(req.params.name, arguments=req.params.arguments),
        )
        return types.ServerResult(result)

    bridge_server.request_handlers[types.CallToolRequest] = call_tool_handler

    @bridge_server.list_resources()
    async def list_resources(req: types.ListResourcesRequest) -> types.ListResourcesResult:
        return await _handle_with_reconnect(
            "resources/list",
            lambda session: session.list_resources(cursor=(req.params.cursor if req.params else None)),
        )

    @bridge_server.list_resource_templates()
    async def list_resource_templates() -> list[types.ResourceTemplate]:
        result = await _handle_with_reconnect(
            "resources/templates/list",
            lambda session: session.list_resource_templates(),
        )
        return result.resourceTemplates

    @bridge_server.read_resource()
    async def read_resource(uri):
        result = await _handle_with_reconnect(
            "resources/read",
            lambda session: session.read_resource(uri),
        )
        return result.contents

    @bridge_server.subscribe_resource()
    async def subscribe_resource(uri):
        await _handle_with_reconnect(
            "resources/subscribe",
            lambda session: session.subscribe_resource(uri),
        )

    @bridge_server.unsubscribe_resource()
    async def unsubscribe_resource(uri):
        await _handle_with_reconnect(
            "resources/unsubscribe",
            lambda session: session.unsubscribe_resource(uri),
        )

    @bridge_server.list_prompts()
    async def list_prompts(req: types.ListPromptsRequest) -> types.ListPromptsResult:
        return await _handle_with_reconnect(
            "prompts/list",
            lambda session: session.list_prompts(cursor=(req.params.cursor if req.params else None)),
        )

    @bridge_server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        return await _handle_with_reconnect(
            "prompts/get",
            lambda session: session.get_prompt(name=name, arguments=arguments),
        )

    @bridge_server.completion()
    async def completion(ref, argument, context):
        result = await _handle_with_reconnect(
            "completion/complete",
            lambda session: session.complete(
                ref=ref,
                argument={argument.name: argument.value},
                context_arguments=(context.arguments if context else None),
            ),
        )
        return result.completion

    @bridge_server.set_logging_level()
    async def set_logging_level(level):
        await _handle_with_reconnect(
            "logging/setLevel",
            lambda session: session.set_logging_level(level),
        )

    async def ping_handler(_: types.PingRequest) -> types.ServerResult:
        result = await _handle_with_reconnect("ping", lambda session: session.send_ping())
        return types.ServerResult(result)

    bridge_server.request_handlers[types.PingRequest] = ping_handler

    return bridge_server


def attach_protocol_bridges(app: FastAPI, server_name: str, serve_protocols: list[str]) -> None:
    if not serve_protocols:
        return

    bridge_server = create_bridge_server(app, server_name=server_name)
    app.state.bridge_server = bridge_server

    if "sse" in serve_protocols:
        sse_transport = SseServerTransport("/sse/messages")
        app.state.sse_bridge_transport = sse_transport

        async def sse_endpoint(request: Request):
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await bridge_server.run(
                    streams[0],
                    streams[1],
                    bridge_server.create_initialization_options(),
                )
            return Response()

        app.add_api_route("/sse", sse_endpoint, methods=["GET"], include_in_schema=False)
        app.mount("/sse/messages", sse_transport.handle_post_message)
        logger.info("Enabled SSE bridge routes for '%s' at /sse and /sse/messages", server_name)

    if "streamable-http" in serve_protocols:
        session_manager = StreamableHTTPSessionManager(
            app=bridge_server,
            stateless=False,
        )
        app.state.streamable_bridge_session_manager = session_manager

        async def streamable_http_asgi(scope, receive, send):
            await session_manager.handle_request(scope, receive, send)

        app.mount("/streamable-http", streamable_http_asgi)
        app.mount("/http-streamable", streamable_http_asgi)
        logger.info(
            "Enabled Streamable HTTP bridge routes for '%s' at /streamable-http (+ alias /http-streamable)",
            server_name,
        )


@contextlib.asynccontextmanager
async def bridge_runtime_context(app: FastAPI):
    streamable_manager = getattr(app.state, "streamable_bridge_session_manager", None)
    if streamable_manager is None:
        yield
        return

    async with streamable_manager.run():
        yield

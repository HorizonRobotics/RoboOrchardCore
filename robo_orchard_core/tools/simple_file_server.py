# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import email.utils
import os
import re
from datetime import datetime
from html import escape  # Used for HTML escaping to prevent XSS
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import AliasChoices, Field
from pydantic_settings import CliImplicitFlag

from robo_orchard_core.tools.cli_bridge import PydanticSettingsTyperAdapter
from robo_orchard_core.utils.cli import SettingConfig

app = FastAPI()

BYTE_RANGE_RE = re.compile(r"bytes=(\d+)-(\d+)?$")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Expose-Headers": "Accept-Ranges, Content-Range",
}

BASE_DIR = os.getenv("ROBO_ORCHARD_SIMPLE_FILE_SERVER_BASE_DIR", os.getcwd())
ALLOW_SYMLINK = False


def resolve_path_from_base(path: str) -> Path | None:
    """Resolve a request path under the configured base directory.

    Args:
        path (str): The relative request path.

    Returns:
        Path | None: The resolved path if it stays within ``BASE_DIR``.
            Returns None when the request escapes the configured base
            directory.
    """
    base_dir = Path(BASE_DIR).resolve()
    request_path = Path(os.path.normpath(base_dir / path.lstrip("/")))
    try:
        request_path.relative_to(base_dir)
    except ValueError:
        return None

    if ALLOW_SYMLINK:
        return request_path

    resolved_path = request_path.resolve(strict=False)
    if resolved_path != base_dir and base_dir not in resolved_path.parents:
        return None
    return resolved_path


def parse_byte_range(byte_range: str) -> tuple[Optional[int], Optional[int]]:
    """Parse the byte range from the Range header.

    Args:
        byte_range (str): The Range header value (e.g., "bytes=0-1023").

    Returns:
        tuple: A tuple of (start, stop) byte positions, where
            either can be None.

    Raises:
        ValueError: If the byte range is invalid.
    """
    # First, strip whitespace from the input string.
    stripped_range = byte_range.strip()
    if not stripped_range:
        return None, None

    # Perform the regex match on the stripped string.
    m = BYTE_RANGE_RE.match(stripped_range)
    if not m:
        raise ValueError("Invalid byte range")

    first, last = [int(x) if x else None for x in m.groups()]
    if last is not None and last < first:  # type: ignore
        raise ValueError("Invalid byte range")
    return first, last


async def copy_byte_range(
    infile, start: int, stop: int, chunk_size: int = 16 * 1024
):
    """Asynchronously read a specific byte range from a file.

    Args:
        infile (str): Path to the file.
        start (int): Starting byte position.
        stop (int): Ending byte position.
        chunk_size (int): Size of chunks to read at a time (default: 16KB).

    Yields:
        bytes: Chunks of the file content within the specified range.
    """
    async with aiofiles.open(infile, "rb") as f:
        await f.seek(start)
        remaining = stop - start + 1
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            chunk = await f.read(to_read)
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def guess_content_type(filepath: str) -> str:
    """Guess the MIME type of a file based on its extension.

    Args:
        filepath (str): Path to the file.

    Returns:
        str: The guessed MIME type (defaults to "application/octet-stream").
    """
    ext = os.path.splitext(filepath)[1].lower()
    mime_types = {
        ".html": "text/html",
        ".css": "text/css",
        ".js": "application/javascript",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".gif": "image/gif",
        ".mcap": "application/octet-stream",
    }
    return mime_types.get(ext, "application/octet-stream")


def get_base_headers(file_stat=None) -> dict:
    """Generate base HTTP headers for responses.

    Args:
        file_stat (os.stat_result, optional): File stat object for
            additional metadata.

    Returns:
        dict: A dictionary of HTTP headers.
    """
    headers = {"Accept-Ranges": "bytes", **CORS_HEADERS}
    if file_stat:
        last_modified = datetime.fromtimestamp(file_stat.st_mtime)
        headers["Last-Modified"] = email.utils.format_datetime(last_modified)
    return headers


def generate_directory_listing(path: str, request_url: str) -> str | None:
    """Generate an HTML directory listing for the given path.

    Args:
        path (str): The relative path within the base directory.
        request_url (str): The full request URL.

    Returns:
        str: HTML content for the directory listing, or None if the path is
        not a directory.
    """
    full_path = resolve_path_from_base(path)
    if full_path is None or not full_path.is_dir():
        return None

    items = sorted(full_path.iterdir(), key=lambda item: item.name)

    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head><title>Directory listing for /{}</title></head>".format(
            escape(path)
        ),
        "<body>",
        "<h1>Directory listing for /{}</h1>".format(escape(path)),
        "<hr>",
        "<ul>",
    ]

    # Add parent directory link (if not root)
    if path:
        parent_path = os.path.dirname(path.rstrip("/"))
        html.append(f'<li><a href="/{parent_path}">../</a></li>')

    # Add file and directory list
    for item in items:
        item_path = os.path.join(path, item.name).lstrip("/")
        if item.is_dir():
            item_display = f"{item.name}/"
        else:
            item_display = item.name
        html.append(
            f'<li><a href="/{escape(item_path)}">{escape(item_display)}</a></li>'  # noqa: E501
        )

    html.extend(
        [
            "</ul>",
            "<hr>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(html)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and return a response with CORS headers.

    Args:
        request (Request): The incoming request.
        exc (HTTPException): The exception to handle.

    Returns:
        Response: A response with the exception details and CORS headers.
    """
    return Response(
        content=exc.detail, status_code=exc.status_code, headers=CORS_HEADERS
    )


@app.options("/{filepath:path}")
async def options_handler(filepath: str):
    """Handle OPTIONS requests for CORS preflight.

    Args:
        filepath (str): The requested file path.

    Returns:
        Response: A 200 response with CORS headers.
    """
    return Response(status_code=200, headers=CORS_HEADERS)


@app.get("/{filepath:path}")
async def serve_file(request: Request, filepath: str):
    """Serve a file or directory listing.

    Args:
        request (Request): The incoming request.
        filepath (str): The requested file path.

    Returns:
        Response: A file response, directory listing, or streaming response
        for partial content.

    Raises:
        HTTPException: If the file or directory is not found, or if the
        range is invalid.
    """
    full_path = resolve_path_from_base(filepath)
    if full_path is None:
        raise HTTPException(status_code=404, detail="File not found")

    # If it's a directory, return a file listing
    if full_path.is_dir():
        html_content = generate_directory_listing(filepath, str(request.url))
        if html_content:
            return HTMLResponse(content=html_content, status_code=200)
        raise HTTPException(
            status_code=404, detail="Directory not found"
        )

    # If it's a file, serve it
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    file_stat = full_path.stat()
    file_size = file_stat.st_size
    headers = get_base_headers(file_stat)

    range_header = request.headers.get("Range")
    if not range_header:
        return FileResponse(
            str(full_path),
            headers=headers,
            media_type=guess_content_type(str(full_path)),
        )

    try:
        start, stop = parse_byte_range(range_header)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid byte range")

    if start is None:
        start = 0
    if stop is None or stop >= file_size:
        stop = file_size - 1

    if start >= file_size:
        raise HTTPException(
            status_code=416, detail="Requested Range Not Satisfiable"
        )

    content_length = stop - start + 1
    headers.update(
        {
            "Content-Range": f"bytes {start}-{stop}/{file_size}",
            "Content-Length": str(content_length),
        }
    )

    async def stream_response():
        async for chunk in copy_byte_range(str(full_path), start, stop):
            yield chunk

    return StreamingResponse(
        stream_response(),
        status_code=206,
        headers=headers,
        media_type=guess_content_type(str(full_path)),
    )


@app.head("/{filepath:path}")
async def head_file(request: Request, filepath: str):
    """Handle HEAD requests to return file metadata.

    Args:
        request (Request): The incoming request.
        filepath (str): The requested file path.

    Returns:
        Response: A response with file metadata headers.

    Raises:
        HTTPException: If the file or directory is not found.
    """
    full_path = resolve_path_from_base(filepath)

    if full_path is None or not full_path.exists() or full_path.is_dir():
        raise HTTPException(status_code=404, detail="Not found")

    file_stat = full_path.stat()
    headers = get_base_headers(file_stat)

    # Add the Content-Length header based on the file size
    headers["Content-Length"] = str(file_stat.st_size)

    return Response(
        status_code=200,
        headers=headers,
        media_type=guess_content_type(str(full_path)),
    )


class FileServerConfig(SettingConfig):
    """Settings model for the ``robo-orchard file-server`` command.

    The command serves files from one local directory over HTTP. Symlink
    traversal is disabled by default; callers must opt in with
    ``allow_symlink`` when they intentionally want linked paths to be served.
    """

    port: Optional[int] = Field(
        default=None,
        description="Port to bind the server to.",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host interface to bind to.",
    )
    directory: str = Field(
        default=".",
        validation_alias=AliasChoices("dir", "directory"),
        description="Directory to serve.",
    )
    allow_symlink: CliImplicitFlag[bool] = Field(
        default=False,
        description="Allow serving files through symlinks.",
    )

    def command_impl(self) -> None:
        """Start the file server using the parsed command-line settings."""
        from robo_orchard_core.utils.network import find_free_port

        port = self.port
        if port is None:
            port = find_free_port()

        start_server(
            host=self.host,
            port=port,
            directory=self.directory,
            allow_symlink=self.allow_symlink,
        )


cli_app = PydanticSettingsTyperAdapter().as_typer(
    FileServerConfig,
    prog="robo-orchard file-server",
    description="Simple HTTP file server.",
)


def start_server(
    host: str, port: int, directory: str, allow_symlink: bool = False
):
    """Start the FastAPI-backed simple file server.

    Args:
        host (str): Host interface to bind to.
        port (int): Port to bind to.
        directory (str): Local directory to serve.
        allow_symlink (bool, optional): Whether resolved symlink targets may
            point outside ``directory``. Default is False.
    """
    target_dir = os.path.abspath(directory)
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    global ALLOW_SYMLINK, BASE_DIR
    BASE_DIR = target_dir
    ALLOW_SYMLINK = allow_symlink
    print(f"Serving files from: {BASE_DIR}")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")

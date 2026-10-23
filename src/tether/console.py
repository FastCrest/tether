"""Rich consoles that tolerate redirected Windows text streams."""

from __future__ import annotations

from typing import IO, Any, cast

from rich.console import Console as RichConsole


class _EncodingSafeStream:
    """Escape only characters the destination cannot encode, before writing."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        encoding = getattr(self._stream, "encoding", None)
        if encoding:
            text = text.encode(encoding, errors="backslashreplace").decode(encoding)
        return self._stream.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def write_human_text(text: str, *, stream: IO[str]) -> None:
    """Write human text without changing its spacing, flushing, or stream policy."""
    _EncodingSafeStream(stream).write(text)


class Console(RichConsole):
    """Keep human diagnostics printable without changing stdout or JSON writers.

    Rich selects ASCII table borders for narrow encodings, but still passes
    literal status symbols and model paths to the stream unchanged. A strict
    cp1252 redirect can therefore abort an otherwise successful export. Wrap
    only Rich's destination, resolving it on each access so capture/redirection
    still works. UTF-8 text is preserved and process-wide stream policies stay
    untouched.
    """

    @property
    def file(self) -> IO[str]:
        return cast(IO[str], _EncodingSafeStream(super().file))

    @file.setter
    def file(self, new_file: IO[str]) -> None:
        self._file = new_file

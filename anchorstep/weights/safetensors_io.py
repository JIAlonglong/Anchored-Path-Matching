"""Raw safetensors header/index inspection and logical state digests."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorEntry:
    key: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    def chunks(self, size: int = 8 * 1024 * 1024):
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            remaining = self.length
            while remaining:
                chunk = stream.read(min(size, remaining))
                if not chunk:
                    raise ValueError(f"truncated safetensors data for {self.key}")
                remaining -= len(chunk)
                yield chunk


def _single_entries(path: Path) -> dict[str, TensorEntry]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")
    data_start = 8 + header_length
    file_size = path.stat().st_size
    result: dict[str, TensorEntry] = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"invalid tensor header for {key}")
        try:
            start, end = value["data_offsets"]
            dtype = str(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor metadata for {key}") from error
        if start < 0 or end < start or data_start + end > file_size:
            raise ValueError(f"invalid data offsets for {key}")
        result[key] = TensorEntry(
            key=key,
            path=path,
            dtype=dtype,
            shape=shape,
            offset=data_start + start,
            length=end - start,
        )
    return result


def tensor_entries(path_or_index: Path) -> dict[str, TensorEntry]:
    """Return validated logical entries from one file or a standard index."""
    path = Path(path_or_index)
    if path.name.endswith(".index.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index requires a nonempty weight_map")
        shards: dict[str, dict[str, TensorEntry]] = {}
        physical_owner: dict[str, str] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_path = path.parent / str(shard_name)
            if not shard_path.is_file():
                raise ValueError(f"missing shard: {shard_name}")
            entries = _single_entries(shard_path)
            for key in entries:
                if key in physical_owner:
                    raise ValueError(
                        f"duplicate tensor key across shards: {key} in "
                        f"{physical_owner[key]} and {shard_name}"
                    )
                physical_owner[key] = str(shard_name)
            shards[str(shard_name)] = entries
        result: dict[str, TensorEntry] = {}
        for key, shard_name in weight_map.items():
            shard = shards.get(str(shard_name))
            if shard is None or key not in shard:
                raise ValueError(f"weight_map entry does not exist in shard: {key}")
            result[str(key)] = shard[str(key)]
        if set(result) != set(physical_owner):
            extra = sorted(set(physical_owner) - set(result))
            raise ValueError(f"shard tensors missing from weight_map: {extra}")
        return result
    if not path.is_file():
        raise ValueError(f"missing safetensors file: {path}")
    return _single_entries(path)


def entries_equal(left: TensorEntry, right: TensorEntry) -> bool:
    if (left.dtype, left.shape, left.length) != (
        right.dtype,
        right.shape,
        right.length,
    ):
        return False
    left_chunks = left.chunks()
    right_chunks = right.chunks()
    return all(a == b for a, b in zip(left_chunks, right_chunks, strict=True))


def tensor_state_digest(path_or_index: Path) -> str:
    """Hash sorted key, dtype, shape, length, and raw bytes without full loading."""
    digest = hashlib.sha256()
    entries = tensor_entries(path_or_index)
    for key in sorted(entries):
        entry = entries[key]
        metadata = json.dumps(
            {
                "key": key,
                "dtype": entry.dtype,
                "shape": list(entry.shape),
                "length": entry.length,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(struct.pack("<Q", len(metadata)))
        digest.update(metadata)
        for chunk in entry.chunks():
            digest.update(chunk)
    return digest.hexdigest()

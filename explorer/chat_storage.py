"""Portable JSONL chat storage, including logs written by older Windows builds."""
import json


def read_events(path):
    events = []
    with open(path, "rb") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                # Previous Windows releases used the system's cp1252 encoding.
                # Decode per record so appending UTF-8 also preserves old logs.
                text = line.decode("cp1252")
            events.append(json.loads(text))
    return events


def write_events(path, events, *, append=False):
    with open(path, "a" if append else "w", encoding="utf-8") as target:
        for event in events:
            target.write(json.dumps(event, ensure_ascii=False) + "\n")

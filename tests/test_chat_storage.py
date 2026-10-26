"""Unicode messages must survive send, streaming, replay and rewind on Windows."""
import builtins
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
from chat_storage import read_events, write_events


class ChatStorageTests(unittest.TestCase):
    def test_unicode_round_trip_with_windows_default_encoding(self):
        original_open = builtins.open

        def windows_open(file, mode='r', *args, **kwargs):
            if 'b' not in mode and 'encoding' not in kwargs:
                kwargs['encoding'] = 'cp1252'
            return original_open(file, mode, *args, **kwargs)

        events = [{'t': 'user', 'text': 'dose × median(‖resid‖) / ‖W_down[:, n]‖ — 你好 🧠'},
                  {'t': 'text', 'text': 'Δ activation: −1 → +1 🧠'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jsonl'
            with patch('builtins.open', side_effect=windows_open):
                write_events(path, events[:1], append=True)
                write_events(path, events[1:], append=True)
                self.assertEqual(read_events(path), events)
                write_events(path, events[:1])
                self.assertEqual(read_events(path), events[:1])
            self.assertIn('‖resid‖', path.read_text(encoding='utf-8'))

    def test_preserves_legacy_windows_records_when_appending_utf8(self):
        old = {'t': 'user', 'text': '“Hello” — café × 0.8'}
        new = {'t': 'text', 'text': '‖resid‖ 🧠'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jsonl'
            path.write_bytes((json.dumps(old, ensure_ascii=False) + '\n').encode('cp1252'))
            write_events(path, [new], append=True)
            self.assertEqual(read_events(path), [old, new])
            write_events(path, read_events(path))
            self.assertEqual([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()], [old, new])


if __name__ == '__main__':
    unittest.main()

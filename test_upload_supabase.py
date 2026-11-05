"""Verify cloud ID allocation requests without sending network traffic."""
import json
import unittest
from urllib.parse import parse_qs, urlsplit
from unittest.mock import MagicMock, patch

import upload_supabase


class CloudIdTests(unittest.TestCase):
    def test_local_ids_are_sent_as_source_ids_and_input_is_unchanged(self):
        records = [
            {"id": 1, "record_time": "2026-11-05T00:00:00Z", "surface_label": None},
            {"id": 1, "record_time": "2026-11-05T00:01:00Z", "surface_label": "asphalt"},
        ]
        response = MagicMock()
        response.__enter__.return_value = response
        with patch.object(upload_supabase, 'urlopen', return_value=response) as request:
            count = upload_supabase.upload_records(records, 'https://example.supabase.co', 'test-key')
        self.assertEqual(count, 2)
        sent = request.call_args.args[0]
        body = json.loads(sent.data)
        self.assertIn("missing=default", sent.get_header("Prefer"))
        self.assertTrue(all('id' not in row for row in body))
        self.assertEqual([row['source_id'] for row in body], [1, 1])
        self.assertIsNone(body[0]['surface_label'])
        self.assertEqual(parse_qs(urlsplit(sent.full_url).query)['on_conflict'], ['source_id,record_time'])
        self.assertTrue(all('source_id' not in row and row['id'] == 1 for row in records))

    def test_retry_uses_the_same_key_and_handles_batches(self):
        records = [{"id": i, "record_time": "2026-11-05T00:00:00Z"} for i in range(1, 4)]
        response = MagicMock()
        response.__enter__.return_value = response
        with patch.dict(upload_supabase.CONFIG, {'batch_size': 2}), patch.object(upload_supabase, 'urlopen', return_value=response) as request:
            upload_supabase.upload_records(records, 'https://example.supabase.co', 'test-key')
            first = [call.args[0].data for call in request.call_args_list]
            request.reset_mock()
            upload_supabase.upload_records(records, 'https://example.supabase.co', 'test-key')
            self.assertEqual(first, [call.args[0].data for call in request.call_args_list])
            self.assertEqual([len(json.loads(body)) for body in first], [2, 1])


if __name__ == '__main__':
    unittest.main()

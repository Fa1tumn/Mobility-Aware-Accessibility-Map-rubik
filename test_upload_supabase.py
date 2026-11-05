"""Verify cloud ID allocation requests without sending network traffic."""
import io
import json
from urllib.error import HTTPError
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


class LegacySurfaceSchemaTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.object(upload_supabase, "_LEGACY_SCHEMA_URLS", set())
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def response(self):
        response = MagicMock()
        response.__enter__.return_value = response
        return response

    def error(self, code, message, status=400):
        return HTTPError("https://example.supabase.co", status, "error", {},
                         io.BytesIO(json.dumps({"code": code, "message": message}).encode()))

    def test_missing_metadata_retries_and_caches_without_changing_records(self):
        records = [{"id": 1, "record_time": "2026-11-05T00:00:00Z",
                    "surface_label": "cement", "surface_confidence": 0.9,
                    **dict.fromkeys(upload_supabase.SURFACE_METADATA_FIELDS, None)}]
        original = json.dumps(records, sort_keys=True)
        for code, message in (
            ("PGRST204", "Could not find the 'area_accept_ratio' column in the schema cache"),
            ("42703", "column observations.area_accept_ratio does not exist"),
        ):
            with self.subTest(code=code):
                upload_supabase._LEGACY_SCHEMA_URLS.clear()
                with patch.object(upload_supabase, "urlopen", side_effect=[
                    self.error(code, message), self.response(), self.response()
                ]) as request:
                    self.assertEqual(upload_supabase.upload_records(records, "https://example.supabase.co", "key"), 1)
                    upload_supabase.upload_records(records, "https://example.supabase.co", "key")
                sent = [json.loads(call.args[0].data)[0] for call in request.call_args_list]
                self.assertIn("area_accept_ratio", sent[0])
                for row in sent[1:]:
                    self.assertTrue(set(upload_supabase.SURFACE_METADATA_FIELDS).isdisjoint(row))
                    self.assertEqual(row["source_id"], 1)
                    self.assertEqual(row["surface_label"], "cement")
                self.assertEqual(json.dumps(records, sort_keys=True), original)

    def test_other_upload_errors_are_not_silenced(self):
        records = [{"id": 1, "area_accept_ratio": 0.5}]
        for code, message, status in (
            ("PGRST204", "Could not find the 'source_id' column", 400),
            ("42501", "permission denied", 403),
        ):
            with self.subTest(code=code), patch.object(upload_supabase, "urlopen", side_effect=
                    self.error(code, message, status)) as request:
                with self.assertRaises(RuntimeError):
                    upload_supabase.upload_records(records, "https://example.supabase.co", "key")
                self.assertEqual(request.call_count, 1)
                self.assertFalse(upload_supabase._LEGACY_SCHEMA_URLS)

    def test_current_schema_keeps_all_metadata(self):
        record = {"id": 1, "model_version": "v3", "area_accept_ratio": 0.5}
        with patch.object(upload_supabase, "urlopen", return_value=self.response()) as request:
            upload_supabase.upload_records([record], "https://example.supabase.co", "key")
        sent = json.loads(request.call_args.args[0].data)[0]
        self.assertEqual(sent["model_version"], "v3")
        self.assertEqual(sent["area_accept_ratio"], 0.5)


if __name__ == '__main__':
    unittest.main()

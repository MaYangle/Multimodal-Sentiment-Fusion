import csv
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import scripts.download_author_raw_cohort as downloader
from scripts.download_author_raw_cohort import canonical_id, read_targets, request_range_ok


class FakeResponse:
    def __init__(self, status_code, headers, body=b"", fail_after=None):
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.fail_after = fail_after
        self.url = "https://drive.usercontent.google.com/download"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_content(self, chunk_size):
        limit = self.fail_after if self.fail_after is not None else len(self.body)
        pos = 0
        while pos < limit:
            end = min(pos + chunk_size, limit)
            yield self.body[pos:end]
            pos = end
        if self.fail_after is not None:
            raise requests.ConnectionError("fixture interruption")


def good_preflight(payload, last_modified="Thu, 01 Jan 1970 00:00:00 GMT", etag='"v1"'):
    return FakeResponse(
        206,
        {
            "Content-Type": "video/mp4",
            "Content-Length": "1",
            "Content-Range": f"bytes 0-0/{len(payload)}",
            "Last-Modified": last_modified,
            "ETag": etag,
        },
        payload[:1],
    )


def good_payload(payload, start, last_modified="Thu, 01 Jan 1970 00:00:00 GMT", etag='"v1"', fail_after=None):
    body = payload[start:]
    return FakeResponse(
        206,
        {
            "Content-Type": "video/mp4",
            "Content-Length": str(len(body)),
            "Content-Range": f"bytes {start}-{len(payload)-1}/{len(payload)}",
            "Last-Modified": last_modified,
            "ETag": etag,
        },
        body,
        fail_after=fail_after,
    )


def make_record(output_path="video_0001/0003.mp4", canonical="video_0001/0003", file_id="drive-one"):
    return {
        "canonical_id": canonical,
        "mode": "train",
        "video_id": canonical.split("/")[0],
        "clip_id": canonical.split("/")[1],
        "file_id": file_id,
        "published_path": f"Raw\\{canonical.split('/')[0]}\\{canonical.split('/')[1]}.mp4",
        "output_path": output_path,
        "competition_source_sha256": None,
        "status": "pending",
        "preflight_attempts": 0,
        "full_payload_attempts": 0,
        "downloaded_payload_bytes": 0,
        "reused_payload_bytes": 0,
    }


def setup_temp_paths(test_case, root):
    return test_case.enterContext(patch.multiple(
        downloader,
        OUTPUT_DIR=root / "raw_cohort",
        MANIFEST_PATH=root / "raw_cohort" / "manifest.json",
        DOC_REPORT=root / "cohort.json",
        RECENT_RECHECK=root / "absent-recheck.json",
        LEGACY_RAW=root / "absent-legacy",
    ))


class AuthorRawCohortTests(unittest.TestCase):
    def test_selects_only_exact_train_valid_ids_even_when_test_rows_interleave(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "meta.csv"
            listing_path = root / "listing.json"
            rows = [
                ["video_0001", "0003", "train", "not-parsed-score"],
                ["video_0009", "9999", "test", "secret-test-score"],
                ["video_0002", "0004", "valid", "also-not-parsed"],
                ["video_0010", "0010", "valid", "must-not-be-selected"],
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video_id", "clip_id", "mode", "label_T"])
                writer.writerows(rows)
            listing = [
                {"path": r"Raw\video_0001\0003.mp4", "id": "drive-train"},
                {"path": r"Raw\video_0002\0004.mp4", "id": "drive-valid"},
                {"path": r"Raw\video_0010\0010.mp4", "id": "drive-valid-2"},
                {"path": r"Raw\video_0009\9999.mp4", "id": "drive-test"},
            ]
            listing_path.write_text(json.dumps(listing), encoding="utf-8")

            targets, pins = read_targets(
                csv_path,
                listing_path,
                expected_split_counts={"train": 1, "valid": 2},
                expected_csv_sha256=None,
                expected_listing_sha256=None,
            )

            self.assertEqual([row["canonical_id"] for row in targets], ["video_0001/0003", "video_0002/0004", "video_0010/0010"])
            self.assertEqual([row["file_id"] for row in targets], ["drive-train", "drive-valid", "drive-valid-2"])
            self.assertEqual(pins["fields_used"], ["mode", "video_id", "clip_id"])
            self.assertTrue(pins["csv_structure_scanned"])
            self.assertFalse(pins["test_rows_selected_or_used"])
            self.assertFalse(pins["score_values_inspected_or_numerically_parsed"])

    def test_exact_versioned_range_is_required(self):
        self.assertTrue(request_range_ok(206, "bytes 0-0/10", "bytes 0-0/10", "Thu, 01 Jan 1970 00:00:00 GMT"))
        self.assertFalse(request_range_ok(200, "bytes 0-0/10", "bytes 0-0/10", "Thu, 01 Jan 1970 00:00:00 GMT"))
        self.assertFalse(request_range_ok(206, "bytes 0-0/11", "bytes 0-0/10", "Thu, 01 Jan 1970 00:00:00 GMT"))
        self.assertFalse(request_range_ok(206, "bytes 0-0/10", "bytes 0-0/10", None))
        self.assertFalse(request_range_ok(206, "bytes 0-0/10", "bytes 0-0/10", "new-version", "old-version"))

    def test_id_path_components_are_safe(self):
        self.assertEqual(canonical_id("video_0001", "0003"), "video_0001/0003")
        with self.assertRaises(ValueError):
            canonical_id("..", "0003")
        with self.assertRaises(ValueError):
            canonical_id("video_0001", "../outside")

    def test_full_payload_hash_receipt_and_cache_reuse(self):
        payload = b"video-payload"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_temp_paths(self, root)
            record = make_record()
            manifest = {"status": "running", "records": [record]}
            calls = []

            def fake_get(url, headers, stream, timeout):
                calls.append(headers.copy())
                if headers["Range"] == "bytes=0-0":
                    return good_preflight(payload)
                return good_payload(payload, 0)

            with patch.object(downloader.requests, "get", side_effect=fake_get):
                status, stop, transferred = downloader.fetch_target(record, manifest)
            output = root / "raw_cohort" / record["output_path"]
            receipt_path = output.with_suffix(output.suffix + ".provenance.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected_sha = hashlib.sha256(payload).hexdigest()
            self.assertEqual((status, stop, transferred), ("complete", False, len(payload)))
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(receipt["sha256"], expected_sha)
            self.assertTrue(receipt["transfer_complete"])
            self.assertTrue(downloader.validate_complete_file(record)[0])
            self.assertEqual(len(calls), 2)

            reuse_record = make_record(output_path="copy/0003.mp4")
            donor = {"path": output, "bytes": len(payload), "sha256": expected_sha, "last_modified": receipt["last_modified"], "source": "fixture-cache"}
            reused_receipt = downloader.use_donor(reuse_record, donor)
            reused_output = root / "raw_cohort" / "copy/0003.mp4"
            self.assertEqual(reused_output.read_bytes(), payload)
            self.assertEqual(reused_receipt["sha256"], expected_sha)
            self.assertEqual(reuse_record["status"], "reused")

    def test_interrupted_payload_resumes_exact_suffix_on_same_version(self):
        payload = b"0123456789abcdef"
        last_modified = "Thu, 01 Jan 1970 00:00:00 GMT"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_temp_paths(self, root)
            record = make_record()
            manifest = {"status": "running", "records": [record]}
            payload_calls = []

            def fake_get(url, headers, stream, timeout):
                range_header = headers["Range"]
                if range_header == "bytes=0-0":
                    return good_preflight(payload, last_modified)
                start = int(range_header.removeprefix("bytes=").split("-")[0])
                payload_calls.append(start)
                if len(payload_calls) == 1:
                    return good_payload(payload, start, last_modified, fail_after=5)
                return good_payload(payload, start, last_modified)

            with patch.object(downloader.requests, "get", side_effect=fake_get):
                status, stop, transferred = downloader.fetch_target(record, manifest)
            output = root / "raw_cohort" / record["output_path"]
            self.assertEqual((status, stop), ("complete", False))
            self.assertEqual(payload_calls, [0, 5])
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(transferred, len(payload))
            self.assertEqual(record["transport_retry_count"], 1)
            receipt = json.loads(output.with_suffix(".mp4.provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["payload_responses"][1]["expected_content_range"], f"bytes 5-{len(payload)-1}/{len(payload)}")

    def test_changed_version_preserves_partial_and_rejects_completion(self):
        payload = b"same-object-bytes"
        old_version = "Thu, 01 Jan 1970 00:00:00 GMT"
        new_version = "Fri, 02 Jan 1970 00:00:00 GMT"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_temp_paths(self, root)
            record = make_record()
            output, partial, state_path, _ = downloader.output_paths(record)
            partial.parent.mkdir(parents=True, exist_ok=True)
            state = downloader.load_partial_state(record, partial, state_path, payload[:1], len(payload), old_version, '"v1"')
            partial.write_bytes(payload[:4])
            downloader.save_partial_state(state_path, partial, state)
            manifest = {"status": "running", "records": [record]}

            with patch.object(downloader.requests, "get", return_value=good_preflight(payload, new_version)) as mocked:
                status, stop, _ = downloader.fetch_target(record, manifest)
            self.assertEqual(status, "partial_integrity_mismatch")
            self.assertTrue(stop)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(partial.read_bytes(), payload[:4])
            self.assertTrue(state_path.exists())
            self.assertFalse(output.exists())

    def test_same_version_partial_is_validated_across_invocations(self):
        payload = b"resume-me-safely"
        version = "Thu, 01 Jan 1970 00:00:00 GMT"
        etag = '"v1"'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_temp_paths(self, root)
            record = make_record()
            _, partial, state_path, _ = downloader.output_paths(record)
            partial.parent.mkdir(parents=True, exist_ok=True)

            first_state = downloader.load_partial_state(record, partial, state_path, payload[:1], len(payload), version, etag)
            partial.write_bytes(payload[:6])
            downloader.save_partial_state(state_path, partial, first_state)

            # A new invocation rechecks the same publisher version and resumes
            # from the exact verified local prefix.
            resumed_state = downloader.load_partial_state(record, partial, state_path, payload[:1], len(payload), version, etag)
            self.assertEqual(resumed_state["partial_bytes"], 6)
            self.assertEqual(partial.read_bytes(), payload[:6])

    def test_corrupt_partial_prefix_is_rejected_even_with_updated_local_hash(self):
        payload = b"trusted-prefix"
        version = "Thu, 01 Jan 1970 00:00:00 GMT"
        etag = '"v1"'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_temp_paths(self, root)
            record = make_record()
            _, partial, state_path, _ = downloader.output_paths(record)
            partial.parent.mkdir(parents=True, exist_ok=True)
            state = downloader.load_partial_state(record, partial, state_path, payload[:1], len(payload), version, etag)
            partial.write_bytes(b"X" + payload[1:5])
            downloader.save_partial_state(state_path, partial, state)

            with self.assertRaises(downloader.StopQueue) as caught:
                downloader.load_partial_state(record, partial, state_path, payload[:1], len(payload), version, etag)
            self.assertEqual(caught.exception.record_status, "partial_integrity_mismatch")
            self.assertTrue(partial.exists())
            self.assertEqual(partial.read_bytes(), b"X" + payload[1:5])

    def test_quota_html_stops_before_next_id(self):
        first_payload = b"first-object"
        targets = [make_record(file_id="drive-one"), make_record("video_0002/0004.mp4", "video_0002/0004", "drive-two")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_temp_paths(self, root)
            calls = []

            def fake_get(url, headers, stream, timeout):
                calls.append((url, headers.copy()))
                if headers["Range"] == "bytes=0-0":
                    return good_preflight(first_payload)
                return FakeResponse(
                    200,
                    {"Content-Type": "text/html", "Content-Length": "64"},
                    b"<html>Download quota exceeded</html>",
                )

            pins = {
                "author_csv_sha256": "fixture-csv",
                "raw_listing_sha256": "fixture-listing",
                "selection_plan_sha256": "fixture-plan",
                "selected_rows": 2,
                "split_counts": {"train": 2, "valid": 0},
            }
            with patch.object(downloader.requests, "get", side_effect=fake_get), patch.object(downloader, "build_donor_map", return_value={}):
                code = downloader.run_queue(targets, pins, {}, False)
            manifest = json.loads((root / "raw_cohort" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 4)
            self.assertEqual(manifest["status"], "stopped_quota")
            self.assertEqual(manifest["records"][0]["status"], "quota_html_on_payload")
            self.assertEqual(manifest["records"][1]["status"], "pending")
            self.assertEqual(len(calls), 2)
            self.assertTrue(all("drive-two" not in url for url, _ in calls))


if __name__ == "__main__":
    unittest.main()

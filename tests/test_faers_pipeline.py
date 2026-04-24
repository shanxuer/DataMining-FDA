import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_faers_pipeline.py"
spec = importlib.util.spec_from_file_location("run_faers_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
sys.modules["run_faers_pipeline"] = pipeline
spec.loader.exec_module(pipeline)


class FaersPipelineTests(unittest.TestCase):
    def test_age_unit_conversion(self):
        self.assertAlmostEqual(pipeline.age_to_years("24", "802"), 2.0)
        self.assertAlmostEqual(pipeline.age_to_years("365.25", "804"), 1.0)
        self.assertEqual(pipeline.age_to_years("999", "801"), None)
        self.assertEqual(pipeline.age_to_years("", "801"), None)

    def test_normalize_drug_name(self):
        self.assertEqual(pipeline.normalize_text("  acetaminophen / codeine  "), "ACETAMINOPHEN CODEINE")
        self.assertEqual(pipeline.make_token("drug", "Bimekizumab 160 mg/mL"), "drug:BIMEKIZUMAB_160_MG_ML")

    def test_label_generation(self):
        flags = {field: "2" for field in pipeline.SERIOUSNESS_FIELDS}
        self.assertEqual(pipeline.label_from_flags(flags), 0)
        flags["seriousnesshospitalization"] = "1"
        self.assertEqual(pipeline.label_from_flags(flags), 1)
        flags["seriousnesshospitalization"] = "2"
        flags["serious"] = "1"
        self.assertEqual(pipeline.label_from_flags(flags), 1)

    def test_target_leakage_not_in_hash_text(self):
        row = {field: "" for field in pipeline.CASE_COLUMNS}
        row.update(
            {
                "quarter": "2025Q1",
                "split": "train",
                "serious": "1",
                "seriousnessdeath": "1",
                "label_serious": "1",
                "patientsex": "2",
                "drug_count": "3",
                "reaction_count": "1",
                "indication_count": "1",
                "text_tokens": "drug:ABC reac:HEADACHE",
            }
        )
        text = pipeline.build_hash_text(row)
        self.assertIn("patientsex:2", text)
        self.assertIn("drug:ABC", text)
        self.assertNotIn("serious", text.lower())
        self.assertNotIn("label_serious", text)

    def test_deleted_id_loader(self):
        path = ROOT / "tests" / "fixtures" / "DELETE_SAMPLE.txt"
        self.assertEqual(pipeline.load_deleted_ids(path), {"123", "456"})

    def test_date_parse(self):
        self.assertEqual(str(pipeline.parse_yyyymmdd("20250131")), "2025-01-31")
        self.assertIsNone(pipeline.parse_yyyymmdd("20250231"))
        self.assertIsNone(pipeline.parse_yyyymmdd(""))

    def test_quarter_split_policy(self):
        self.assertEqual(pipeline.split_for_quarter("2025Q1"), "train")
        self.assertEqual(pipeline.split_for_quarter("2025Q2"), "train")
        self.assertEqual(pipeline.split_for_quarter("2025Q3"), "train")
        self.assertEqual(pipeline.split_for_quarter("2025Q4"), "q4_pending")

    def test_q4_assigns_validation_and_test_by_receivedate(self):
        rows = [
            {"receivedate": "20251004", "safetyreportid": "4"},
            {"receivedate": "20251001", "safetyreportid": "1"},
            {"receivedate": "20251003", "safetyreportid": "3"},
            {"receivedate": "20251002", "safetyreportid": "2"},
        ]
        counts = pipeline.assign_q4_splits(rows)
        self.assertEqual(counts["valid"], 2)
        self.assertEqual(counts["test"], 2)
        self.assertEqual([row["split"] for row in rows], ["test", "valid", "test", "valid"])

    def test_q4_ties_sort_by_safetyreportid_and_bad_dates_last(self):
        rows = [
            {"receivedate": "bad", "safetyreportid": "1"},
            {"receivedate": "20251001", "safetyreportid": "20"},
            {"receivedate": "20251001", "safetyreportid": "10"},
            {"receivedate": "", "safetyreportid": "2"},
        ]
        pipeline.assign_q4_splits(rows)
        self.assertEqual(rows[2]["split"], "valid")
        self.assertEqual(rows[1]["split"], "valid")
        self.assertEqual(rows[0]["split"], "test")
        self.assertEqual(rows[3]["split"], "test")


if __name__ == "__main__":
    unittest.main()

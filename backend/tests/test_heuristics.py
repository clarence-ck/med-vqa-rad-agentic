"""
Unit tests for the rule-based parts of the pipeline and the evaluation scorer.

Run from backend/:  python -m unittest discover tests
No API calls or model downloads are made.
"""
import os
import sys
import unittest

os.environ.setdefault("MEDVQA_DISABLE_FILE_LOGS", "1")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.question_analyzer import QuestionAnalyzer  # noqa: E402
from agents.reasoning_agent import ReasoningAgent  # noqa: E402
from agents.verification_agent import VerificationAgent  # noqa: E402
from agents.orchestrator import _normalize_final_answer  # noqa: E402
from utils import scoring  # noqa: E402
from utils.answer_utils import leading_yes_no  # noqa: E402


def _style(question: str):
    return QuestionAnalyzer._infer_question_style(None, question).get("question_type")


class QuestionStyleTests(unittest.TestCase):
    def test_ct_substring_does_not_trigger_modality(self):
        for question in [
            "Is there a fracture?",
            "What structure is shown?",
            "Is there evidence of small bowel obstruction?",
            "Is there mass effect?",
            "Are the bones intact?",
        ]:
            self.assertNotEqual(_style(question), "modality_identification", question)

    def test_real_modality_questions_still_detected(self):
        for question in ["Is this a CT scan?", "What modality is used?", "Is this an x-ray?", "Is this an MRI?"]:
            self.assertEqual(_style(question), "modality_identification", question)

    def test_side_substring_does_not_trigger_location(self):
        self.assertNotEqual(_style("Is this considered normal?"), "location")
        self.assertEqual(_style("Which side is the lesion on?"), "location")


class YesNoNormalisationTests(unittest.TestCase):
    def test_leading_yes_no(self):
        self.assertEqual(leading_yes_no("No."), "no")
        self.assertEqual(leading_yes_no("yes, there is"), "yes")
        for text in ["normal", "nodule", "none", "nodular opacities", "not seen"]:
            self.assertIsNone(leading_yes_no(text), text)

    def test_agents_do_not_fold_normal_into_no(self):
        self.assertNotEqual(ReasoningAgent._normalize_yes_no_answer(None, "nodule"), "no")
        self.assertNotEqual(_normalize_final_answer("nodule", {"question_type": "yes_no"}, {}), "no")


class VerifierCountingTests(unittest.TestCase):
    def test_counting_heuristic_no_longer_raises(self):
        result = VerificationAgent._format_heuristics(
            VerificationAgent.__new__(VerificationAgent),
            "3", {"question_subtype": "counting_exact"}, "how many nodules?",
            count_estimate=1, count_confidence="high", count_method="visual_estimate",
        )
        self.assertFalse(result["checks"]["answer_strength_appropriate"])


class ScorerTests(unittest.TestCase):
    def lenient_similarity(self, *_):
        return 0.99  # simulate an embedding model that rates everything as similar

    def test_laterality_mismatch_is_wrong_even_if_embeddings_agree(self):
        self.assertFalse(scoring.answers_match("left lung", "right lung", False, None, self.lenient_similarity))
        self.assertFalse(scoring.answers_match("left cerebral hemisphere", "right temporal lobe", False, None, self.lenient_similarity))

    def test_short_answers_need_content_match(self):
        self.assertFalse(scoring.answers_match("liver", "brain", False, None, lambda *_: 0.91))
        self.assertTrue(scoring.answers_match("Axial T2-weighted brain MRI", "t2-mri", False, None, lambda *_: 0.0))
        self.assertTrue(scoring.answers_match("CT scan", "computed tomography", False, None, lambda *_: 0.0))
        self.assertTrue(scoring.answers_match("both kidneys", "kidneys", False, None, lambda *_: 0.0))

    def test_negated_prediction_cannot_match_positive_finding(self):
        self.assertFalse(scoring.answers_match(
            "No ventricular abnormality seen; the lateral ventricles appear normal",
            "lateral and third ventricular hydrocephalus", False, None, self.lenient_similarity))

    def test_close_ended(self):
        self.assertTrue(scoring.is_close_ended("yes"))
        self.assertTrue(scoring.is_close_ended("normal"))
        self.assertFalse(scoring.is_close_ended("nodular opacities"))
        self.assertTrue(scoring.answers_match("No.", "no", True))
        self.assertFalse(scoring.answers_match("normal", "no", True))
        self.assertTrue(scoring.answers_match("abnormal", "abnormal", True))
        self.assertFalse(scoring.answers_match("normal", "abnormal", True))

    def test_counting(self):
        self.assertTrue(scoring.answers_match("two nodules", "2", False, "counting"))
        self.assertFalse(scoring.answers_match("3", "2", False, "counting"))

    def test_agreement_is_symmetric(self):
        self.assertTrue(scoring.answers_agree("kidneys", "both kidneys", False))
        self.assertFalse(scoring.answers_agree("yes", "no", True))


class AttemptSelectionTests(unittest.TestCase):
    def test_later_answer_must_be_strictly_better_verified(self):
        from agents.agentic_loop import select_best_attempt
        partial = {"verification_status": "partially_verified", "confidence_label": "Medium"}
        verified = {"verification_status": "verified", "confidence_label": "High"}
        low = {"verification_status": "low_confidence", "confidence_label": "Low"}
        self.assertEqual(select_best_attempt([partial, partial]), 0)   # tie keeps the original
        self.assertEqual(select_best_attempt([low, verified]), 1)      # improvement replaces it
        self.assertEqual(select_best_attempt([verified, partial]), 0)  # regression is ignored


class ConsistencyLevelTests(unittest.TestCase):
    def test_levels(self):
        from agents.agentic_loop import consistency_level
        self.assertEqual(consistency_level(2, 2), "high")
        self.assertEqual(consistency_level(1, 2), "medium")
        self.assertEqual(consistency_level(0, 2), "low")
        self.assertEqual(consistency_level(0, 0), "not_run")


if __name__ == "__main__":
    unittest.main()

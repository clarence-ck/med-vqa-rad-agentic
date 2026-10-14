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

    def test_findings_on_a_named_modality_are_not_modality_questions(self):
        for question in [
            "is a mass present in this xray?",
            "can the optic nerve be visualized in this mri image?",
            "are there any skeletal joints in the ct?",
        ]:
            self.assertNotEqual(_style(question), "modality_identification", question)

    def test_side_substring_does_not_trigger_location(self):
        self.assertNotEqual(_style("Is this considered normal?"), "location")
        self.assertEqual(_style("Which side is the lesion on?"), "location")

    def test_either_or_questions_are_not_yes_no(self):
        for question in [
            "is this supratentorial or infratentorial?",
            "are the structures in the pancreas cystic or solid?",
            "is this a t1 weighted, t2 weighted, or flair image?",
            "is the mass solid or cystic?",
        ]:
            self.assertNotIn(_style(question), {"yes_no", "presence_absence"}, question)
        for question in ["Is there a fracture or dislocation?", "Are there nodules or masses?", "Can you see a mass or lesion?"]:
            self.assertIn(_style(question), {"yes_no", "presence_absence"}, question)


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


class ControllerPathTests(unittest.TestCase):
    def test_refused_abstain_is_not_an_abstention(self):
        from agents.agentic_loop import controller_path
        actions = ["analyze_question", "look", "answer", "finish(abstain)", "check", "finish"]
        self.assertEqual(controller_path(actions, abstained=False), "direct")
        self.assertEqual(controller_path(actions + ["finish(abstain)"], abstained=True), "abstained")
        self.assertEqual(controller_path(["look", "answer", "check", "look(left kidney)", "answer(guided)"]), "relook")
        self.assertEqual(controller_path(["look", "answer", "check", "answer(guided)", "check"]), "reanswer")


class ConsistencyLevelTests(unittest.TestCase):
    def test_levels(self):
        from agents.agentic_loop import consistency_level
        self.assertEqual(consistency_level(2, 2), "high")
        self.assertEqual(consistency_level(1, 2), "medium")
        self.assertEqual(consistency_level(0, 2), "low")
        self.assertEqual(consistency_level(0, 0), "not_run")

class ImportTests(unittest.TestCase):
    def test_every_module_imports(self):
        """Catches removed or renamed functions that another module still imports."""
        import importlib
        for module in ("agents.agentic_loop", "agents.orchestrator", "agents.workflow_baseline", "agents.baseline",
                       "routes.analyze", "routes.dataset", "main"):
            importlib.import_module(module)


class RunSelectionTests(unittest.TestCase):
    def test_current_run_is_latest_completed_unless_overridden(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock
        from utils import runs
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runs, "RUNS_DIR", Path(tmp)),                 mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEDVQA_RUN", None)
            self.assertIsNone(runs.current_run_id())
            runs.write_run("2026-01-01_0900", {"status": "completed"})
            runs.write_run("2026-02-01_0900", {"status": "running"})
            self.assertEqual(runs.current_run_id(), "2026-01-01_0900")  # an unfinished run is not shown
            self.assertEqual(runs.latest_unfinished_run_id(), "2026-02-01_0900")
            os.environ["MEDVQA_RUN"] = "2026-02-01_0900"
            self.assertEqual(runs.current_run_id(), "2026-02-01_0900")


def _png(width=400, height=300):
    from io import BytesIO
    from PIL import Image
    out = BytesIO()
    Image.new("L", (width, height), 128).save(out, format="PNG")
    return out.getvalue()


class ZoomTests(unittest.TestCase):
    def test_parse_box(self):
        from utils.image_zoom import parse_box
        self.assertEqual(parse_box([0.1, 0.2, 0.5, 0.6]), (0.1, 0.2, 0.5, 0.6))
        self.assertEqual(parse_box([10, 20, 50, 60]), (0.1, 0.2, 0.5, 0.6))  # percent
        for bad in (None, [0.5, 0.5, 0.2, 0.9], [1, 2, 3], "box", [0, 0, 500, 500]):
            self.assertIsNone(parse_box(bad), bad)

    def test_crop_is_enlarged_and_kept_inside_the_image(self):
        from io import BytesIO
        from PIL import Image
        from utils.image_zoom import zoom, OUT_LONG_SIDE, MIN_SIDE
        png, box = zoom(_png(), [0.9, 0.9, 0.95, 0.95])  # tiny box in the corner
        x0, y0, x1, y1 = box
        self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1)
        self.assertGreaterEqual(round(x1 - x0, 3), MIN_SIDE)
        self.assertEqual(max(Image.open(BytesIO(png)).size), OUT_LONG_SIDE)
        self.assertIsNone(zoom(b"not an image", [0.1, 0.1, 0.5, 0.5]))


class BlindCheckTests(unittest.TestCase):
    def _check(self, status="verified", action="accept"):
        return {"verification_status": status, "confidence_label": "High", "recommended_action": action,
                "issues_found": [], "relook_focus": ""}

    def test_agreement_keeps_the_verdict(self):
        from agents.agentic_loop import apply_blind_agreement
        check = apply_blind_agreement(self._check(), "Yes, there is a fracture.", {"status": "success", "answer": "yes"},
                                      {"question_type": "yes_no", "expected_answer_format": "yes_no"})
        self.assertEqual((check["verification_status"], check["blind_agreement"]), ("verified", True))

    def test_disagreement_blocks_verified_and_hides_the_blind_answer(self):
        from agents.agentic_loop import apply_blind_agreement
        check = apply_blind_agreement(self._check(), "no", {"status": "success", "answer": "Yes, a left rib fracture"},
                                      {"question_type": "yes_no", "expected_answer_format": "yes_no",
                                       "target_entity": "ribs"})
        self.assertEqual(check["verification_status"], "partially_verified")
        self.assertEqual((check["recommended_action"], check["relook_focus"]), ("relook", "ribs"))
        self.assertFalse(check["blind_agreement"])
        self.assertNotIn("rib fracture", " ".join(check["issues_found"]))  # controller never sees the blind answer

    def test_failed_blind_reading_changes_nothing(self):
        from agents.agentic_loop import apply_blind_agreement
        check = apply_blind_agreement(self._check(), "no", {"status": "error", "answer": ""}, {})
        self.assertEqual((check["verification_status"], check["blind_agreement"]), ("verified", None))


class LookZoomTests(unittest.TestCase):
    def _run(self, locate_reply):
        from unittest import mock
        from agents import look_agent
        replies = [locate_reply, '{"is_radiology_image": true, "modality": "X-ray", "visual_summary": "s"}']
        calls = []

        def run(messages, generation_kwargs):
            calls.append(messages[0])
            return {"replies": [mock.Mock(text=replies[len(calls) - 1], meta={})]}

        with mock.patch.object(look_agent, "OpenAIChatGenerator", mock.Mock(return_value=mock.Mock(run=run))), \
                mock.patch.object(look_agent, "Secret", mock.Mock()):
            result = look_agent.LookAgent().run(image_bytes=_png(), question="fracture?", focus="left ribs")["result"]
        return result, calls

    def test_located_region_is_sent_as_an_extra_zoomed_image(self):
        result, calls = self._run('{"found": true, "box": [0.6, 0.3, 0.9, 0.7]}')
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[1].images), 2)  # full image + enlarged crop
        self.assertIsNotNone(result["zoom_box"])

    def test_unlocated_region_falls_back_to_a_plain_relook(self):
        result, calls = self._run('{"found": false}')
        self.assertEqual(len(calls[1].images), 1)
        self.assertIsNone(result["zoom_box"])



class LoopBlindReadingTests(unittest.TestCase):
    """One faked episode through the real loop: answer 'no', blind reader says 'yes'."""

    def test_disagreement_is_recorded_but_never_shown_to_the_controller(self):
        from unittest import mock
        from agents import agentic_loop as loop
        seen = {}

        class FakeAgent:
            def __init__(self, tools, **_):
                self.tools = {t.name: t.function for t in tools}

            def run(self, messages):
                for name in ("analyze_question", "look", "answer"):
                    self.tools[name]()
                seen["first_check"] = self.tools["check"]()
                self.tools["answer"](guidance="re-examine")
                seen["second_check"] = self.tools["check"]()
                self.tools["finish"]()
                return {"messages": []}

        def fake(result):
            return mock.Mock(return_value=mock.Mock(run=mock.Mock(return_value={"result": result})))

        blind = mock.Mock(return_value={"status": "success", "answer": "Yes, clearly visible", "explanation": "e"})
        with mock.patch.object(loop, "Agent", FakeAgent), \
                mock.patch.object(loop, "OpenAIChatGenerator", mock.Mock()), \
                mock.patch.object(loop, "Secret", mock.Mock()), \
                mock.patch.object(loop, "SELF_CONSISTENCY_SAMPLES", 0), \
                mock.patch.object(loop, "blind_read", blind), \
                mock.patch.object(loop, "QuestionAnalyzer", fake({"status": "success", "question_type": "yes_no",
                                                                 "expected_answer_format": "yes_no"})), \
                mock.patch.object(loop, "LookAgent", fake({"status": "success", "modality": "CT", "visual_summary": "v"})), \
                mock.patch.object(loop, "ReasoningAgent", fake({"status": "success", "answer": "no", "explanation": "e"})), \
                mock.patch.object(loop, "VerificationAgent", fake({"verification_status": "verified", "confidence_label": "High",
                                                                  "recommended_action": "accept", "checks": {}})):
            result = loop.run_agentic_loop(b"img", "is there a fracture?")
        blind.assert_called_once()  # one blind reading per episode, reused by every check
        for text in seen.values():
            self.assertIn("verification_status=partially_verified", text)
            self.assertNotIn("clearly visible", text)
        self.assertEqual(result["verification_status"], "partially_verified")
        self.assertEqual(result["blind_reading"], {"answer": "Yes, clearly visible", "agrees_with_final": False})
        self.assertIn("Blind Reading", [step["agent"] for step in result["agent_trace"]])


if __name__ == "__main__":
    unittest.main()


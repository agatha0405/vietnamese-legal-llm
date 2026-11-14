import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app import extract_content, prepare_messages  # noqa: E402
from app.grounding import LegalIndex  # noqa: E402
from app.inference import CLARIFY_VEHICLE, NO_SANCTION_SOURCE, chat_config, sanction_reply  # noqa: E402


class DummyTokenizer:
    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True):
        return list(" ".join(m["content"] for m in msgs))


def messages(question):
    return [{"role": "system", "content": "test"}, {"role": "user", "content": question}]


class ChatInferenceTest(unittest.TestCase):
    def test_unqualified_fines_ask_for_vehicle(self):
        for question in ("Vượt đèn đỏ phạt bao nhiêu?", "Chạy quá tốc độ bị phạt thế nào?"):
            with self.subTest(question=question):
                self.assertEqual(sanction_reply(messages(question)), (CLARIFY_VEHICLE, "clarify_vehicle"))

    def test_known_vehicle_and_follow_up_do_not_invent_sanctions(self):
        question = "Không đội mũ bảo hiểm khi lái xe máy phạt bao nhiêu?"
        expected = NO_SANCTION_SOURCE, "unverified_sanction"
        self.assertEqual(sanction_reply(messages(question)), expected)
        history = messages("Vượt đèn đỏ phạt bao nhiêu?") + [
            {"role": "assistant", "content": CLARIFY_VEHICLE},
            {"role": "user", "content": "Xe máy ạ"},
        ]
        self.assertEqual(sanction_reply(history), expected)

    def test_non_sanction_questions_continue_to_model(self):
        for question in ("Bằng lái A1 dùng cho xe nào?", "Giới hạn tốc độ là gì?",
                         "Trốn thuế bị phạt bao nhiêu?", "Đăng ký kết hôn ở đâu?"):
            with self.subTest(question=question):
                self.assertIsNone(sanction_reply(messages(question)))

    def test_profile_preserves_model_identity_and_legacy_configuration(self):
        qwen = {"system_prompt": "original", "model_revision": "pinned", "generation": {
            "max_new_tokens": 256, "do_sample": False, "eos_token_id": [151645, 151643],
        }}
        self.assertEqual(chat_config(qwen, "legacy"), qwen)
        new = chat_config(qwen, "concise")
        self.assertEqual(new["model_revision"], qwen["model_revision"])
        self.assertEqual(new["generation"]["eos_token_id"], qwen["generation"]["eos_token_id"])
        self.assertEqual(new["generation"]["max_new_tokens"], 96)
        self.assertEqual(qwen["generation"]["max_new_tokens"], 256)

    def test_prepare_messages_preserves_gradio_v6_multimodal_history(self):
        t1 = "Theo Điều 999 Luật Trật tự, an toàn giao thông đường bộ, vượt đèn đỏ bị phạt bao nhiêu?"
        index = LegalIndex()
        d1 = index.resolve([{"role": "system", "content": "sys"}, {"role": "user", "content": t1}])
        self.assertEqual(d1.kind, "clarify")

        history = [
            {"role": "user", "content": [{"type": "text", "text": t1}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"Fine-tuned · 1.25 giây\n\n{d1.render()}"}]},
        ]
        self.assertEqual(extract_content(history[0]["content"]), t1)
        prepared = prepare_messages(DummyTokenizer(), "sys", "Xe máy", history)
        self.assertEqual(len(prepared), 4)
        self.assertEqual(prepared[1]["content"], t1)
        self.assertIn(CLARIFY_VEHICLE, prepared[2]["content"])

        d2 = index.resolve(prepared)
        self.assertEqual(d2.kind, "grounded")
        self.assertEqual(d2.rule, "signal-moto")
        self.assertIn("không có Điều 999", d2.text)
        self.assertNotIn("Nguồn:", d2.render())

        switch_history = history + [
            {"role": "user", "content": [{"type": "text", "text": "Xe máy"}]},
            {"role": "assistant", "content": [{"type": "text", "text": d2.render()}]},
        ]
        prepared_car = prepare_messages(DummyTokenizer(), "sys", "Còn ô tô thì sao?", switch_history)
        d3_car = index.resolve(prepared_car)
        self.assertEqual(d3_car.kind, "grounded")
        self.assertEqual(d3_car.rule, "signal-car")
        self.assertIn("18.000.000", d3_car.render())

        prepared_exp = prepare_messages(DummyTokenizer(), "sys", "Xe máy có được lên cao tốc không?", switch_history)
        self.assertIsNone(index.resolve(prepared_exp))
        aug_exp = index.augment_messages(prepared_exp)
        self.assertEqual(len(aug_exp), 2)
        self.assertNotIn("vượt đèn đỏ", aug_exp[0]["content"].lower())


if __name__ == "__main__":
    unittest.main()

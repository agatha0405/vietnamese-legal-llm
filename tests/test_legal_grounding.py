from datetime import date
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.grounding import LegalIndex, official_url, unverified_sanction_output  # noqa: E402


class LegalGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = LegalIndex()

    def lookup(self, question, history=(), **kwargs):
        return self.index.resolve([
            {"role": "system", "content": "test"}, *history, {"role": "user", "content": question},
        ], **kwargs)

    def test_verified_cases_return_the_expected_rule(self):
        cases = json.loads(Path(__file__).with_name("chatbot_grounding_cases.json").read_bytes())
        for case in cases:
            if case["group"] == "answerable":
                with self.subTest(case=case["id"]):
                    decision = self.lookup(case["question"])
                    self.assertEqual(decision.kind, "grounded")
                    if "rule" in case:
                        self.assertEqual(decision.rule, case["rule"])
                    self.assertTrue(decision.url.startswith("https://"))

    def test_sanctions_regression_uses_one_lookup_and_citation(self):
        cases = json.loads(Path(__file__).with_name("chatbot_sanctions_cases.json").read_bytes())
        for case in cases:
            with self.subTest(case=case["id"]):
                decision = self.lookup(case["question"])
                self.assertEqual(decision.kind, case["kind"])
                self.assertEqual(decision.rule, case.get("rule"))
                rendered = decision.render()
                self.assertNotIn("Nguồn: https://", rendered)
                self.assertNotIn("Căn cứ:", rendered)
                self.assertNotIn("Mốc áp dụng", rendered)
                self.assertNotIn("Văn bản sửa đổi", rendered)
                if decision.kind == "grounded":
                    self.assertTrue(decision.url.startswith("https://"))
                    self.assertTrue(decision.citation)
                    self.assertIn("Nguồn: https://", decision.render(include_source=True))
        self.assertEqual(self.lookup("Xe đạp điện vượt đèn đỏ phạt như nào?").kind, "no_source")

    def test_unverified_amounts_are_rejected_only_in_model_output(self):
        for answer in ("Phạt từ 1.000.000–1.500.000 đồng", "Phạt hai triệu", "1.5tr", "2 triệu VND", "Bị trừ 4 điểm"):
            self.assertTrue(unverified_sanction_output(answer), answer)
        for answer in ("Tốc độ tối đa 30 km/h", "Ngưỡng hạng A1 là 125 cm³", "Theo khoản 4 Điều 11"):
            self.assertFalse(unverified_sanction_output(answer), answer)
        self.assertFalse(official_url("https://chinhphu.vn.example.org/source"))

    def test_vehicle_follow_up_uses_the_previous_violation(self):
        question = "Vượt đèn đỏ phạt bao nhiêu?"
        clarification = self.lookup(question)
        self.assertEqual(clarification.kind, "clarify")
        decision = self.lookup("Tôi đi xe máy.", [
            {"role": "user", "content": question}, {"role": "assistant", "content": clarification.text},
        ])
        self.assertEqual(decision.rule, "signal-moto")
        self.assertIn("4.000.000", decision.text)

    def test_different_vehicle_amounts_and_conditions(self):
        car = self.lookup("Ô tô vượt đèn đỏ không gây tai nạn thì bị phạt bao nhiêu?")
        self.assertIn("18.000.000", car.text)
        self.assertEqual(self.lookup("Xe máy vượt đèn đỏ gây tai nạn phạt bao nhiêu?").kind, "no_source")
        self.assertEqual(self.lookup("Xe máy chuyên dùng vượt đèn đỏ phạt bao nhiêu?").kind, "no_source")
        self.assertEqual(self.lookup("Xe máy rẽ phải khi vượt đèn đỏ bị phạt bao nhiêu?").kind, "no_source")
        self.assertEqual(self.lookup("Tôi đi xe máy nhưng không vượt đèn đỏ, tại sao bị phạt?").kind, "no_source")
        self.assertEqual(self.lookup("Ô tô hoặc xe máy vượt đèn đỏ phạt bao nhiêu?").kind, "clarify")
        self.assertEqual(self.lookup("Ô tô sử dụng điện thoại bị phạt bao nhiêu?").kind, "clarify")

    def test_facts_follow_up_can_unlock_the_provision(self):
        question = "Ô tô sử dụng điện thoại bị phạt bao nhiêu?"
        clarification = self.lookup(question)
        history = [{"role": "user", "content": question}, {"role": "assistant", "content": clarification.text}]
        self.assertEqual(self.lookup("Có, xe đang di chuyển và tôi dùng tay cầm điện thoại.", history).rule, "phone-car")
        self.assertEqual(self.lookup("Không, xe đã đỗ.", history).kind, "no_source")

    def test_date_filters_pre_effective_and_unreviewed_law(self):
        question = "Xe máy vượt đèn đỏ bị phạt bao nhiêu"
        for day in ("31/12/2024", "15/11/2026", "31/02/2026", "12/11"):
            with self.subTest(day=day):
                self.assertEqual(self.lookup(f"{question} ngày {day}?").kind, "no_source")
        old = self.lookup(f"{question} ngày 14/08/2026?")
        new = self.lookup(f"{question} ngày 15/08/2026?")
        self.assertEqual(old.rule, new.rule)
        self.assertEqual(old.target_date, date(2026, 8, 14))
        self.assertNotIn("14/08/2026", old.render())
        self.assertIn("14/08/2026", old.render(include_source=True))
        self.assertEqual(self.lookup(f"{question} hiện nay?", today=date(2026, 11, 15)).kind, "no_source")

    def test_scope_does_not_treat_unrelated_topics_as_a_traffic_violation(self):
        self.assertEqual(self.lookup("Ở Nhật Bản, xe máy vượt đèn đỏ bị phạt bao nhiêu?").kind, "outside")
        self.assertEqual(self.lookup("Tôi kê khai thuế thu nhập ở đâu?").kind, "outside")
        self.assertIsNone(self.lookup("Mô tả đường ưu tiên."))

    def test_expanded_sanctions_cover_common_violations_and_points_provision(self):
        wrong_moto = self.lookup("Đi xe máy ngược chiều trên đường một chiều bị xử phạt như thế nào?")
        self.assertEqual(wrong_moto.kind, "grounded")
        self.assertEqual(wrong_moto.rule, "wrong-way-moto")
        self.assertIn("4.000.000 đến 6.000.000 đồng", wrong_moto.text)
        self.assertNotIn("Căn cứ:", wrong_moto.render())
        self.assertNotIn("Nguồn:", wrong_moto.render())
        self.assertIn("Điều 7 Nghị định 168/2024/NĐ-CP", wrong_moto.citation)

        wrong_moto_pts = self.lookup("Đi xe máy ngược chiều trên đường một chiều bị phạt bao nhiêu và trừ mấy điểm?")
        self.assertEqual(wrong_moto_pts.rule, "wrong-way-moto")
        self.assertIn("trừ 2 điểm", wrong_moto_pts.text)

        wrong_car = self.lookup("Ô tô đi ngược chiều trên đường một chiều phạt bao nhiêu?")
        self.assertEqual(wrong_car.rule, "wrong-way-car")
        self.assertIn("18.000.000 đến 20.000.000 đồng", wrong_car.text)

        wrong_hwy = self.lookup("Ô tô đi ngược chiều trên đường cao tốc phạt bao nhiêu?")
        self.assertEqual(wrong_hwy.rule, "wrong-way-expressway-car")
        self.assertIn("30.000.000 đến 40.000.000 đồng", wrong_hwy.text)

        phone_moto = self.lookup("Đang chạy xe máy mà dùng tay cầm điện thoại bị phạt bao nhiêu?")
        self.assertEqual(phone_moto.rule, "phone-moto")
        self.assertIn("800.000 đến 1.000.000 đồng", phone_moto.text)

        passenger_helmet = self.lookup("Đi xe máy chở người ngồi sau không đội mũ bảo hiểm bị phạt bao nhiêu?")
        self.assertEqual(passenger_helmet.rule, "helmet-passenger-moto")
        self.assertIn("400.000 đến 600.000 đồng", passenger_helmet.text)

        turn_moto = self.lookup("Xe máy chuyển hướng không bật xi nhan bị phạt bao nhiêu?")
        self.assertEqual(turn_moto.rule, "turn-signal-moto")
        self.assertIn("600.000 đến 800.000 đồng", turn_moto.text)

        lane_car = self.lookup("Ô tô chuyển làn đường không có tín hiệu báo trước bị phạt thế nào?")
        self.assertEqual(lane_car.rule, "lane-change-car")
        self.assertIn("600.000 đến 800.000 đồng", lane_car.text)

        seatbelt = self.lookup("Lái xe ô tô không thắt dây đai an toàn bị phạt bao nhiêu?")
        self.assertEqual(seatbelt.rule, "seatbelt-car")
        self.assertIn("800.000 đến 1.000.000 đồng", seatbelt.text)

        sidewalk = self.lookup("Xe máy đi trên vỉa hè bị phạt bao nhiêu?")
        self.assertEqual(sidewalk.rule, "sidewalk-moto")
        self.assertIn("4.000.000 đến 6.000.000 đồng", sidewalk.text)

        expressway = self.lookup("Đi xe máy vào đường cao tốc bị phạt bao nhiêu?")
        self.assertEqual(expressway.rule, "expressway-moto")
        self.assertIn("4.000.000 đến 6.000.000 đồng", expressway.text)

        alcohol = self.lookup("Lái ô tô có nồng độ cồn bị phạt bao nhiêu?")
        self.assertEqual(alcohol.rule, "alcohol-car")
        self.assertIn("30.000.000 đến 40.000.000 đồng", alcohol.text)
        self.assertNotIn("Điều 6", alcohol.render())
        alcohol_src = self.lookup("Lái ô tô có nồng độ cồn bị phạt bao nhiêu, kèm căn cứ?")
        self.assertIn("Căn cứ:", alcohol_src.render())

        points_q = self.lookup("Giấy phép lái xe có bao nhiêu điểm và khi nào được phục hồi điểm?")
        self.assertEqual(points_q.rule, "license-points-d58")
        self.assertIn("12 điểm", points_q.text)

    def test_three_turn_clarification_chain_and_grounded_vehicle_switch(self):
        t1_q = "Dùng điện thoại bị phạt bao nhiêu?"
        d1 = self.lookup(t1_q)
        self.assertEqual(d1.kind, "clarify")

        history = [{"role": "user", "content": t1_q}, {"role": "assistant", "content": d1.render()}]
        d2 = self.lookup("Ô tô", history)
        self.assertEqual(d2.kind, "clarify")

        history.extend([{"role": "user", "content": "Ô tô"}, {"role": "assistant", "content": d2.render()}])
        d3 = self.lookup("Có, đang chạy và dùng tay cầm", history)
        self.assertEqual(d3.kind, "grounded")
        self.assertEqual(d3.rule, "phone-car")
        self.assertEqual(self.lookup("Có", history).rule, "phone-car")

        moto = self.lookup("Xe máy vượt đèn đỏ phạt bao nhiêu?")
        self.assertEqual(moto.rule, "signal-moto")
        car_switch = self.lookup("Còn ô tô thì sao?", [
            {"role": "user", "content": "Xe máy vượt đèn đỏ phạt bao nhiêu?"},
            {"role": "assistant", "content": moto.render()},
        ])
        self.assertEqual(car_switch.kind, "grounded")
        self.assertEqual(car_switch.rule, "signal-car")
        self.assertIn("18.000.000", car_switch.text)

    def test_fake_and_repealed_citations_and_multiturn_preservation(self):
        bare_fake = self.lookup("Điều 999 quy định gì?")
        self.assertEqual(bare_fake.kind, "invalid_citation")
        self.assertEqual(bare_fake.render(), "Bạn vui lòng cho biết tên văn bản pháp luật. Tôi chưa xác minh được điều khoản này.")

        fake_q = "Theo Điều 999 Luật Trật tự, an toàn giao thông đường bộ, vượt đèn đỏ bị phạt bao nhiêu?"
        d1 = self.lookup(fake_q)
        self.assertEqual(d1.kind, "clarify")
        self.assertIn("không có Điều 999", d1.text)
        self.assertIn("89 điều", d1.text)
        self.assertIn("Bạn đi loại phương tiện nào", d1.text)

        d2 = self.lookup("Xe máy", [
            {"role": "user", "content": fake_q},
            {"role": "assistant", "content": d1.render()},
        ])
        self.assertEqual(d2.kind, "grounded")
        self.assertEqual(d2.rule, "signal-moto")
        self.assertIn("không có Điều 999", d2.text)
        self.assertIn("4.000.000 đến 6.000.000 đồng", d2.text)

        standalone_fake = self.lookup("Nội dung Điều 999 Luật Trật tự, an toàn giao thông đường bộ quy định gì?")
        self.assertEqual(standalone_fake.kind, "invalid_citation")
        self.assertIn("không có Điều 999", standalone_fake.text)

        bad_168 = self.lookup("Điều 80 Nghị định 168/2024/NĐ-CP quy định gì?")
        self.assertEqual(bad_168.kind, "invalid_citation")
        self.assertIn("55 điều", bad_168.text)

        repealed = self.lookup("Theo Nghị định 100/2019/NĐ-CP thì vượt đèn đỏ phạt bao nhiêu?")
        self.assertIn("hết hiệu lực", repealed.text)

    def test_no_false_positive_output_filter_and_rag_context_injection(self):
        master_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "legal_qa_master.jsonl"
        for line in master_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            self.assertFalse(unverified_sanction_output(row["answer"]), row["id"])

        oos_path = Path(__file__).resolve().parents[1] / "data" / "out_of_scope_questions.json"
        for q in json.loads(oos_path.read_text(encoding="utf-8")):
            with self.subTest(out_of_scope=q):
                self.assertEqual(self.lookup(q).kind, "outside")

        augmented = self.index.augment_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Mô tả đường ưu tiên."},
        ])
        self.assertIn("nhường đường", augmented[0]["content"])
        self.assertNotIn("khoản 4 Điều 2", augmented[0]["content"])

        augmented_with_source = self.index.augment_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Mô tả đường ưu tiên và nêu rõ căn cứ."},
        ])
        self.assertIn("khoản 4 Điều 2", augmented_with_source[0]["content"])

        augmented_moto = self.index.augment_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Xe máy là gì?"},
        ])
        self.assertIn("hai hoặc ba bánh", augmented_moto[0]["content"])

        switched = self.index.augment_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Đi xe máy ngược chiều trên đường một chiều bị xử phạt như thế nào?"},
            {"role": "assistant", "content": "Phạt 4.000.000 đến 6.000.000 đồng."},
            {"role": "user", "content": "Mô tả đường ưu tiên."},
        ])
        self.assertEqual(len(switched), 2)
        self.assertIn("nhường đường", switched[0]["content"])
        self.assertNotIn("ngược chiều", switched[0]["content"])

        followup_rag = self.index.augment_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Khi gặp sự cố kỹ thuật buộc phải đỗ xe chiếm một phần đường xe chạy thì phải báo hiệu như thế nào?"},
            {"role": "assistant", "content": "Phải bật đèn khẩn cấp hoặc đặt biển cảnh báo."},
            {"role": "user", "content": "Nếu sự cố đó xảy ra trong hầm đường bộ và xe không di chuyển được thì phải làm gì?"},
        ])
        self.assertEqual(len(followup_rag), 4)
        self.assertIn("hầm đường bộ", followup_rag[0]["content"])

    def test_minimal_bug_fixes_topic_switch_missing_facts_routing_and_concise_output(self):
        # Test 1: Chuyển chủ đề sau câu hỏi vượt đèn đỏ không được lặp lại mức phạt cũ
        t1_q = "Xe máy vượt đèn đỏ phạt bao nhiêu tiền?"
        d1 = self.lookup(t1_q)
        self.assertEqual(d1.kind, "grounded")
        self.assertEqual(d1.rule, "signal-moto")
        self.assertIn("4.000.000 đến 6.000.000 đồng", d1.render())
        self.assertNotIn("Nguồn:", d1.render())
        self.assertNotIn("Mốc áp dụng", d1.render())

        t2_q = "Xe máy có được lên cao tốc không?"
        history_12 = [{"role": "user", "content": t1_q}, {"role": "assistant", "content": d1.render()}]
        self.assertIsNone(self.lookup(t2_q, history_12))
        hits_exp = self.index.retrieve(t2_q, top_k=2)
        self.assertTrue(hits_exp)
        self.assertEqual(hits_exp[0]["id"], "vn-road-55-25-3")
        aug_exp = self.index.augment_messages([
            {"role": "system", "content": "sys"},
            *history_12,
            {"role": "user", "content": t2_q},
        ])
        self.assertEqual(len(aug_exp), 2)
        self.assertNotIn("vượt đèn đỏ", aug_exp[0]["content"].lower())
        self.assertNotIn("vượt đèn đỏ", aug_exp[-1]["content"].lower())

        # Yêu cầu cụ thể căn cứ ở lượt tiếp nối vẫn tra cứu được nội bộ
        src_followup = self.lookup("Cho biết căn cứ và nguồn?", history_12)
        self.assertEqual(src_followup.kind, "grounded")
        self.assertIn("Căn cứ:", src_followup.render())
        self.assertIn("Nguồn: https://", src_followup.render())

        # Test 3: Thiếu tình tiết ở câu hỏi tai nạn hoặc vi phạm chung -> ưu tiên hỏi làm rõ
        acc = self.lookup("Xe ô tô gây tai nạn bị phạt thế nào?")
        self.assertEqual(acc.kind, "clarify")
        self.assertIn("hành vi", acc.render())
        self.assertNotIn("triệu đồng", acc.render())
        generic_viol = self.lookup("Đi xe máy vi phạm giao thông bị phạt bao nhiêu?")
        self.assertEqual(generic_viol.kind, "clarify")
        resolved_viol = self.lookup("Vượt đèn đỏ", [
            {"role": "user", "content": "Đi xe máy vi phạm giao thông bị phạt bao nhiêu?"},
            {"role": "assistant", "content": generic_viol.render()},
        ])
        self.assertEqual(resolved_viol.rule, "signal-moto")

        # Test 4: Câu hỏi độc lập về khái niệm không bị nhiễm chéo hay ghép đoạn luật sai
        one_way = "Đường một chiều là gì?"
        two_way = "Đường hai chiều là gì?"
        self.assertIsNone(self.lookup(one_way))
        self.assertEqual(self.index.retrieve(one_way), [])
        self.assertEqual(self.index.retrieve(two_way), [])
        aug_two = self.index.augment_messages([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": one_way},
            {"role": "assistant", "content": "Đường một chiều chỉ cho phép đi theo một chiều."},
            {"role": "user", "content": two_way},
        ])
        self.assertEqual(len(aug_two), 2)
        self.assertNotIn("một chiều", aug_two[0]["content"].lower())

        # Bug 4 filter_output: bỏ cụm "Theo khoản ... Điều ..." nếu người dùng không yêu cầu căn cứ
        filtered = self.index.filter_output(
            [{"role": "user", "content": "Đường ưu tiên là gì?"}],
            "Theo khoản 4 Điều 2 Luật Trật tự, an toàn giao thông đường bộ 36/2024/QH15, đường ưu tiên là đường được nhường đường.",
        )
        self.assertFalse(filtered.startswith("Theo khoản"))
        self.assertTrue(filtered.startswith("Đường ưu tiên"))
        kept_when_asked = self.index.filter_output(
            [{"role": "user", "content": "Đường ưu tiên là gì theo điều nào?"}],
            "Theo khoản 4 Điều 2 Luật 36/2024/QH15, đường ưu tiên là đường được nhường đường.",
        )
        self.assertTrue(kept_when_asked.startswith("Theo khoản 4 Điều 2"))


if __name__ == "__main__":
    unittest.main()

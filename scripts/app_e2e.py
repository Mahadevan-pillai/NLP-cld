"""Task 16 end-to-end application checks + latency measurements (REAL frozen models, REAL MongoDB Atlas).

    python scripts/app_e2e.py

This is APPLICATION testing, not research evaluation: the inputs are SAFE SYNTHETIC examples (an invented
sentence and a generated meme with a printed caption), no dataset meme is used, and nothing changes any model.
Records written to MongoDB are tagged source="e2e_test" (developer decision, Task 16).

Writes (numbers / IDs only): results/tables/app/app_performance.csv, results/tables/app/app_e2e_checks.csv
"""

import importlib.util
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = REPO / "results" / "tables" / "app"
TEXTS = ["Kal exam hai aur maine kuch nahi padha, bhai help karo", "The weather is very pleasant today",
         "Mera dost kal cricket match dekhne gaya tha", "Is movie ka gaana bahut accha hai", "Office meeting kal subah 10 baje hai",
         "Ghar ka khana sabse best hota hai"]
CAPTIONS = [("JAB EXAM KAL HO", "aur syllabus pura baki ho"), ("MONDAY MORNING", "office jaane ka mann nahi"),
            ("CHAI WITH FRIENDS", "best feeling ever")]


def meme(top: str, bottom: str, color=(70, 110, 160)) -> bytes:
    img = Image.new("RGB", (800, 600), (235, 225, 200))
    d = ImageDraw.Draw(img)
    d.rectangle((250, 150, 550, 450), fill=color)
    try:
        font = ImageFont.truetype("arialbd.ttf", 54)
    except OSError:
        font = ImageFont.load_default()
    d.text((60, 30), top, font=font, fill=(0, 0, 0))
    d.text((60, 500), bottom, font=font, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    perf, checks = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)[:200]})
        print(("PASS " if ok else "FAIL ") + name, detail if not ok else "", flush=True)

    # ---- start-up: imports, audit (checksums vs the Task 13 freeze), model loading
    t0 = time.perf_counter()
    from src.application import artifacts
    from src.application.history import HistoryService
    from src.application.service import AnalysisService
    from src.preprocessing import PreprocessingConfig, preprocess
    perf.append({"measure": "import application modules", "seconds": round(time.perf_counter() - t0, 2), "runs": 1})
    t0 = time.perf_counter()
    rows = artifacts.audit(verify_checksums=True)
    perf.append({"measure": "start-up audit incl. SHA-256 of all final artifacts", "seconds": round(time.perf_counter() - t0, 2), "runs": 1})
    bad = [r["component"] for r in rows if r["status"] != "available"]
    check("all components available and checksums match the Task 13 freeze", not bad, bad)
    svc = AnalysisService(audit_rows=rows)
    t0 = time.perf_counter()
    warm = svc.warm_up()
    perf.append({"measure": "model loading (warm-up: T05+B07, I03+SigLIP, M04, EasyOCR)", "seconds": round(time.perf_counter() - t0, 2), "runs": 1})
    for k, v in warm.items():
        perf.append({"measure": f"  loading: {k}", "seconds": v, "runs": 1})
    spec = importlib.util.spec_from_file_location("fe", REPO / "scripts" / "final_evaluation.py")
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    from src.utils.config import load_config
    fe.verify_freeze(load_config())
    check("verify_freeze(): Task 13 models, code and configs unchanged", True)
    history = HistoryService()
    st = history.status()
    check("MongoDB reachable", st["ok"], st["message"])

    # ---- TEXT end-to-end: service -> preprocessing -> T05 -> explanation -> moderation -> history -> feedback
    r = svc.analyze_text(TEXTS[0])
    keys = {"schema_version", "analysis_type", "status", "input", "prediction", "models", "moderation", "explanation",
            "model_versions", "timings_ms", "disclaimer"}
    check("text: result schema complete", keys <= set(r) and r["status"] == "ok", r.get("error"))
    check("text: FINAL_TEXT_MODEL T05 used", r["prediction"]["model_version"] == "T05" and r["model_versions"] == {"text_model": "T05"})
    check("text: normalised text == Task 6 preprocess(hinglish_normalized)",
          r["input"]["normalized_text"] == preprocess(TEXTS[0], "hinglish_normalized", PreprocessingConfig()))
    check("text: explanation computed (Integrated Gradients)", r["explanation"]["text"]["status"] == "ok")
    check("text: moderation is ALLOW / WARNING / HUMAN_REVIEW", r["moderation"]["recommendation"] in ("ALLOW", "WARNING", "HUMAN_REVIEW"))
    if st["ok"]:
        s = history.save(r, consent=True, source="e2e_test")
        got = history.get(s["analysis_id"])
        check("text: saved to MongoDB and read back", s["saved"] and got["ok"] and got["item"]["source"] == "e2e_test", s["message"])
        fb = history.add_feedback(s["analysis_id"], "correct", "Task 16 end-to-end test")
        check("text: feedback stored", fb["ok"] and history.get(s["analysis_id"])["feedback"], fb["message"])
        check("text: no consent -> nothing saved", not history.save(r, consent=False)["saved"])
        print("  text analysis_id", s["analysis_id"])

    # ---- MEME end-to-end: upload -> validation -> OCR -> preprocessing -> 3 models -> moderation -> explain -> history
    data = meme(*CAPTIONS[0])
    m = svc.analyze_meme(data)
    o = m["input"]["ocr"]
    check("meme: analysis ok", m["status"] == "ok", m.get("error"))
    check("meme: OCR read the caption", o["status"] == "ok" and "EXAM" in o["text"].upper(), o["text"])
    check("meme: multimodal route; T05 / I03 / M04 all ran",
          o["route"] == "multimodal" and m["model_versions"] == {"text_model": "T05", "image_model": "I03", "multimodal_model": "M04"})
    check("meme: primary prediction is the multimodal model", m["prediction"]["model_version"] == "M04")
    check("meme: image model did not use text", m["models"]["image"]["used_input"] == "image only (no text)")
    check("meme: moderation + agreement present", m["moderation"]["recommendation"] in ("ALLOW", "WARNING", "HUMAN_REVIEW")
          and m["modality"]["agreement"] in ("AGREE", "PARTIAL DISAGREEMENT", "DISAGREE"))
    t0 = time.perf_counter()
    e = svc.explain_meme(data, o["text"])
    perf.append({"measure": "meme explanation, first call (includes one-off set-up)", "seconds": round(time.perf_counter() - t0, 2), "runs": 1})
    times = []
    for top, bottom in CAPTIONS[1:]:
        d2 = meme(top, bottom)
        t0 = time.perf_counter()
        svc.explain_meme(d2, f"{top} {bottom}")
        times.append(time.perf_counter() - t0)
    perf.append({"measure": "meme explanation on demand, warm (occlusion 9 copies + IG on OCR text)",
                 "seconds": round(sum(times) / len(times), 2), "runs": len(times)})
    check("meme: explanation (image heatmap + text attribution)", e["image"]["status"] == "ok" and e["text"]["status"] == "ok")
    if st["ok"]:
        s = history.save(m, consent=True, source="e2e_test")
        got = history.get(s["analysis_id"])
        item = got["item"] or {}
        check("meme: saved without image binary (metadata only)", s["saved"] and "image_reference" not in item.get("input", {})
              and set(item.get("input", {}).get("image_metadata", {})) <= {"width", "height", "format", "mode", "size_bytes"}, s["message"])
        check("meme: OCR status and moderation stored", item.get("ocr", {}).get("route") == "multimodal" and item.get("moderation"))
        fb = history.add_feedback(s["analysis_id"], "correct", "Task 16 end-to-end test")
        check("meme: feedback stored", fb["ok"], fb["message"])
        print("  meme analysis_id", s["analysis_id"])
    blank = svc.analyze_meme(meme("", "", (120, 120, 120)))
    check("meme without text: image-only fallback (limited)", blank["status"] == "limited"
          and blank["prediction"]["model_version"] == "I03", blank["input"].get("ocr", {}).get("route"))
    check("corrupt upload rejected safely", svc.analyze_meme(b"\x89PNG broken")["error"]["category"] == "invalid_input")

    # ---- latency (warm models)
    times = []
    for t in TEXTS[1:]:
        t0 = time.perf_counter()
        svc.analyze_text(t, explain=False)
        times.append(time.perf_counter() - t0)
    perf.append({"measure": "text analysis (prediction + second opinion + moderation)", "seconds": round(sum(times) / len(times), 3), "runs": len(times)})
    times = []
    for t in TEXTS[1:]:
        t0 = time.perf_counter()
        svc.explain_text(t)
        times.append(time.perf_counter() - t0)
    perf.append({"measure": "text explanation (T05 Integrated Gradients, 32 steps)", "seconds": round(sum(times) / len(times), 3), "runs": len(times)})
    ocr_t, total_t = [], []
    for top, bottom in CAPTIONS[1:]:
        res = svc.analyze_meme(meme(top, bottom))
        ocr_t.append(res["timings_ms"]["ocr"] / 1000)
        total_t.append(res["timings_ms"]["total"] / 1000)
    perf.append({"measure": "meme OCR (EasyOCR, CPU)", "seconds": round(sum(ocr_t) / len(ocr_t), 2), "runs": len(ocr_t)})
    perf.append({"measure": "meme analysis total (validation + OCR + 3 models + ablation + moderation)", "seconds": round(sum(total_t) / len(total_t), 2), "runs": len(total_t)})

    # ---- Streamlit smoke test with the REAL services (no consent -> no extra records)
    try:
        from streamlit.testing.v1 import AppTest

        sys.path.insert(0, str(REPO / "app"))
        at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=300)
        t0 = time.perf_counter()
        at.run()
        perf.append({"measure": "Streamlit first page render in this process (models already loaded; includes the "
                                "cached start-up audit)", "seconds": round(time.perf_counter() - t0, 2), "runs": 1})
        failed = []
        for page in ("pages/home.py", "pages/analyze_text.py", "pages/analyze_meme.py", "pages/history.py",
                     "pages/dashboard.py", "pages/about.py"):
            at.switch_page(page).run()
            if at.exception:
                failed.append(page)
        check("Streamlit: all six pages render with real services", not failed, failed)
        at.switch_page("pages/analyze_text.py").run()
        at.text_area[0].input(TEXTS[1])
        at.button[0].click().run()
        text = " ".join(x.value for x in at.markdown)
        check("Streamlit: Analyze Text shows FINAL ANALYSIS with the real model", "FINAL ANALYSIS" in text and not at.exception)
        at.switch_page("pages/analyze_meme.py").run()
        at.file_uploader[0].set_value(("upload.png", meme(*CAPTIONS[2]), "image/png")).run()
        at.button[0].click().run()                                     # Analyze, no consent -> nothing stored
        text = " ".join(x.value for x in at.markdown)
        check("Streamlit: Analyze Meme shows OCR text, three models and moderation with the real models",
              "FINAL ANALYSIS" in text and "MODEL COMPARISON" in text and any("CHAI" in t.value.upper() for t in at.text)
              and not at.exception)
        next(b for b in at.button if b.label == "Explain prediction").click().run()
        check("Streamlit: Explain prediction renders the heatmap", len(at.get("image")) >= 2 and not at.exception)
    except ImportError:
        check("Streamlit smoke test", False, "streamlit not installed")

    pd.DataFrame(perf).to_csv(OUT / "app_performance.csv", index=False, lineterminator="\n")
    pd.DataFrame(checks).to_csv(OUT / "app_e2e_checks.csv", index=False, lineterminator="\n")
    print(pd.DataFrame(perf).to_string(index=False))
    n_fail = sum(not c["passed"] for c in checks)
    print(f"{len(checks) - n_fail}/{len(checks)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

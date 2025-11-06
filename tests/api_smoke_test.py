# tests/api_smoke_test.py
import argparse
import requests
import time
from collections import Counter

BASE = "http://127.0.0.1:8000"

def start_session(user_id: int):
    r = requests.post(f"{BASE}/session/start", json={"user_id": user_id})
    r.raise_for_status()
    js = r.json()
    # örnek: {"session_id": "...", "user": {...}}
    return js["session_id"]

def get_next(user_id: int, session_id: str):
    r = requests.get(f"{BASE}/rl/next", params={"user_id": user_id, "session_id": session_id})
    r.raise_for_status()
    js = r.json()
    # ÖRN beklenen: {"word": {"id": 123, "level": "B2", "l2_text": "..."}, "state_dim": 13, "eps": 0.12}
    return js

def post_answer(payload: dict):
    r = requests.post(f"{BASE}/rl/answer", json=payload)
    r.raise_for_status()
    return r.json()  # örn: {"reward": 1.23, "loss": 0.31, "target_level": "B2"}

def run_loop(user_id: int, steps: int):
    sid = start_session(user_id=user_id)
    hist = Counter()
    rewards = []
    for t in range(steps):
        js = get_next(user_id, sid)
        lvl = js["bucket_level"]
        wid = js["word_id"]
        hist[lvl] += 1

        options = js["options"]
        correct_opt = next(opt for opt in options if opt["is_correct"])
        wrong_opts = [opt for opt in options if not opt["is_correct"]]

        # Basit bir cevap stratejisi: hedefe yakınsa doğru, uzaksa yanlış olasılığı yükselsin
        is_correct = True
        if lvl in ("A1","A2"):
            is_correct = (t % 5 == 0)   # arada doğru
        elif lvl == "C1":
            is_correct = (t % 3 != 0)   # bazen yanlış
        # B1/B2'de çoğunlukla doğru

        if is_correct or not wrong_opts:
            selected_text = correct_opt["text"]
        else:
            selected_text = wrong_opts[0]["text"]

        answer_payload = {
            "user_id": user_id,
            "session_id": sid,
            "question_id": js["question_id"],
            "word_id": wid,
            "selected_text": selected_text,
            "response_ms": 2500,
            "bucket_level": lvl,
            "action": js["action"],
        }

        ans = post_answer(answer_payload)
        rewards.append(ans.get("reward", 0.0))

        # opsiyonel: yeni target_level dönerse izle
        if (t+1) % 50 == 0:
            print(f"[{t+1}] ort. reward={sum(rewards)/len(rewards):.3f}  dağılım={dict(hist)}")

        time.sleep(0.01)

    total = sum(hist.values())
    print("\n--- SONUÇ ---")
    for k in ["A1","A2","B1","B2","C1"]:
        print(k, hist[k], f"({hist[k]/total:.1%})")
    print("Ort. reward:", round(sum(rewards)/len(rewards), 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=1, help="Hangi kullanıcı ID'siyle test edileceği")
    parser.add_argument("--steps", type=int, default=200, help="Kaç adım çalıştırılacağı")
    args = parser.parse_args()

    run_loop(user_id=args.user, steps=args.steps)

import requests
import random
import time

BASE_URL = "http://127.0.0.1:8000"

def start_session(user_id=1):
    r = requests.post(f"{BASE_URL}/session/start", json={"user_id": user_id})
    r.raise_for_status()
    return r.json()

def get_question(user_id, session_id):
    r = requests.get(f"{BASE_URL}/rl/next", params={"user_id": user_id, "session_id": session_id})
    r.raise_for_status()
    return r.json()

def send_answer(user_id, session_id, question, chosen_text, response_ms=2500):
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "question_id": question["question_id"],
        "word_id": question["word_id"],
        "selected_text": chosen_text,
        "response_ms": response_ms,
        "bucket_level": question["bucket_level"],
        "action": question["action"]
    }
    r = requests.post(f"{BASE_URL}/rl/answer", json=payload)
    r.raise_for_status()
    return r.json()

def main():
    print("🧠 Başlatılıyor...")
    sess = start_session(user_id=1)
    session_id = sess["session_id"]
    print(f"Yeni oturum: {session_id}")

    for i in range(5):  # 5 tur test
        print(f"\n=== TUR {i+1} ===")
        q = get_question(1, session_id)
        print(f"Soru: {q['prompt']} -> seçenekler: {[opt['text'] for opt in q['options']]}")
        correct_option = next(opt for opt in q["options"] if opt["is_correct"])
        wrong_option = next(opt for opt in q["options"] if not opt["is_correct"])

        # %70 doğru, %30 yanlış simülasyonu
        selected = correct_option if random.random() < 0.7 else wrong_option
        print(f"Seçilen: {selected['text']} (doğru mu? {selected['is_correct']})")

        t0 = time.time()
        ans = send_answer(1, session_id, q, selected["text"], int((time.time()-t0)*1000+random.randint(1500,3500)))
        print(f"Yanıt sonucu -> reward: {ans['reward']:.2f}, loss: {ans['loss']}, correct: {ans['correct']}")

    print("\n✅ Test tamamlandı.")

if __name__ == "__main__":
    main()

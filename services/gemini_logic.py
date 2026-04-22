import os
import google.generativeai as genai

def get_rice_feedback(disease_name: str):
    if disease_name.lower() == "healthy":
        return "Padi Anda sehat. Jaga kebersihan saluran irigasi dan gunakan pupuk berimbang."

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"Berperanlah sebagai ahli pertanian. Berikan saran teknis untuk penyakit padi '{disease_name}' "
            f"dalam 3 poin: 1. Penyebab, 2. Gejala Utama, 3. Solusi Teknis. "
            f"Jawab langsung ke poinnya, jangan pakai kalimat pembuka."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Saran penanganan sedang disiapkan, silakan cek riwayat nanti."
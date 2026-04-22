# services/storage.py
import os
import requests

def upload_to_cpanel(filename, content, content_type):
    url = os.getenv("CPANEL_UPLOAD_URL")
    try:
        files = {"file": (filename, content, content_type)}
        res = requests.post(url, files=files, timeout=8)
        if res.status_code == 200:
            return res.json().get("file_name")
    except Exception as e:
        print(f"Upload Error: {e}")
    return filename

# services/database.py
import os
import mysql.connector

def save_history(user_id, image_name, result, feedback):
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()
        import json
        query = """INSERT INTO history_prediksi 
                   (user_id, image_name, predicted_class, confidence, all_probabilities, feedback) 
                   VALUES (%s, %s, %s, %s, %s, %s)"""
        cursor.execute(query, (
            user_id, image_name, result['predicted_class'], 
            result['confidence'], json.dumps(result['all_probabilities']), feedback
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")
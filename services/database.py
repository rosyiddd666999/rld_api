import os
import mysql.connector
import json

def get_db_conn():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        connect_timeout=10
    )

def get_or_create_user(google_id, email, name):
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # Use INSERT IGNORE to skip if email already exists
    cursor.execute("""
        INSERT IGNORE INTO users (google_id, email, name)
        VALUES (%s, %s, %s)
    """, (google_id, email, name))
    
    conn.commit()
    
    # Always fetch the user after insert
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return user['id']

def save_prediction(user_id, image_name, result, feedback, alamat=None):
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        query = """INSERT INTO history_prediksi 
                   (user_id, image_name, predicted_class, confidence, all_probabilities, feedback, alamat) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(query, (
            user_id, image_name, result['predicted_class'], 
            result['confidence'], json.dumps(result['all_probabilities']), 
            feedback, alamat
        ))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def fetch_history_by_user(user_id=None):
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    try:
        # SQL JOIN untuk mengambil data history + nama user
        query = """
            SELECT h.*, u.name as user_name 
            FROM history_prediksi h
            JOIN users u ON h.user_id = u.id
        """
        if user_id:
            query += " WHERE h.user_id = %s"
            query += " ORDER BY h.created_at DESC"
            cursor.execute(query, (user_id,))
        else:
            query += " ORDER BY h.created_at DESC"
            cursor.execute(query)
            
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
import os
import psycopg2
import psycopg2.extras
import json

def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        port=os.getenv("DB_PORT", 5432),
        connect_timeout=10
    )

def get_or_create_user(google_id, email, name):
    conn = get_db_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        INSERT INTO users (google_id, email, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (email) DO NOTHING
    """, (google_id, email, name))

    conn.commit()

    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    return user['id']

def save_prediction(user_id, image_name, result, feedback, alamat=None, latitude=None, longitude=None):
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        query = """INSERT INTO history_prediksi 
                   (user_id, image_name, predicted_class, confidence, all_probabilities, feedback, alamat, latitude, longitude) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(query, (
            user_id, image_name, result['predicted_class'],
            result['confidence'], json.dumps(result['all_probabilities']),
            feedback, alamat, latitude, longitude
        ))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def fetch_history_by_user(user_id=None):
    conn = get_db_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        query = """
            SELECT h.*, u.name as user_name 
            FROM history_prediksi h
            JOIN users u ON h.user_id = u.id
        """
        if user_id:
            query += " WHERE u.google_id = %s ORDER BY h.created_at DESC"
            cursor.execute(query, (str(user_id),))
        else:
            query += " ORDER BY h.created_at DESC"
            cursor.execute(query)

        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def update_user_profile(google_id, alamat, latitude, longitude):
    conn = get_db_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor.execute("""
            UPDATE users
            SET alamat = %s, latitude = %s, longitude = %s
            WHERE google_id = %s
            RETURNING id, name, email, alamat, latitude, longitude
        """, (alamat, latitude, longitude, google_id))
        conn.commit()
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

def get_user_profile(google_id):
    conn = get_db_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor.execute("""
            SELECT id, name, email, alamat, latitude, longitude
            FROM users
            WHERE google_id = %s
        """, (google_id,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

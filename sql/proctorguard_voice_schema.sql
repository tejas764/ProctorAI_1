CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enrollment_questions (
    question_id TEXT PRIMARY KEY,
    question_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enrollment_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    feature_json TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(question_id) REFERENCES enrollment_questions(question_id),
    UNIQUE(user_id, question_id)
);

CREATE TABLE IF NOT EXISTS speaker_profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    enrollment_complete INTEGER NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS runtime_voice_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp_s REAL NOT NULL,
    similarity REAL,
    drift REAL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

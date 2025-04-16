PORT="${PORT:-8080}"
CORS_ALLOW_ORIGIN=http://localhost:5173 uvicorn open_webui.main:app --port 8080 --host 0.0.0.0 --forwarded-allow-ips '*' --reload
# ChatGPT
Paying around w/ scraping, vectoring > composing

source .venv/bin/activate

uvicorn [main]:app --reload
uvicorn gen-compose:app --reload
uvicorn gen-compose:app --reload --port 8001

lsof -i :8000
kill <PID>
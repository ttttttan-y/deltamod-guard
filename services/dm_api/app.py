from fastapi import FastAPI
app = FastAPI(title="DeltaMod-Guard API (placeholder)")

@app.get("/health")
def health():
    return {"status":"ok"}

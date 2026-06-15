import uvicorn

if __name__ == "__main__":
    uvicorn.run("gateway.main:app", host="0.0.0.0", port=9010, reload=True)

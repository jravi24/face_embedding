from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import httpx

app = FastAPI()

class APIRequest(BaseModel):
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, str]] = None
    body: Optional[Any] = None  # Could be dict or list or primitive

@app.post("/proxy")
async def proxy(request: APIRequest):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method=request.method.upper(),
                #url=request.url,
                url="https://11d1-34-67-196-219.ngrok-free.app/generateEmbedding",
                headers=request.headers,
                params=request.params,
                json=request.body  # Automatically sets content-type to application/json
            )

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if "application/json" in response.headers.get("content-type", "") else response.text
            }
        except httpx.RequestError as exc:
            return {"error": f"An error occurred while requesting {exc.request.url!r}.",
                   "detail":str(exc)}
        except Exception as e:
            return {"error": str(e)}

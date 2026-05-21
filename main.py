from fastapi import FastAPI
from app.api.routes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TravelAgentAPI",
    description="A FastAPI service for travel planning and booking workflows.",
    version="0.1.0",
)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://travelagentapi-dtagfjaxbvhybmeg.canadacentral-01.azurewebsites.net"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "TravelAgentAPI"}

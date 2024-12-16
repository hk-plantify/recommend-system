from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from exception import ApplicationException

def global_exception_handler(app: FastAPI):
    @app.exception_handler(ApplicationException)
    async def handle_application_exception(request: Request, exc: ApplicationException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": exc.status_code, "message": exc.detail, "data": None},
        )
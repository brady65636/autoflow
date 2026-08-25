from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from .work_order_api import create_app  # noqa: E402

app = create_app()

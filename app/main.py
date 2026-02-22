import io
import os
import uuid
from pathlib import Path
from typing import List, Optional

import requests
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, VectorParams
from qdrant_client.models import PointStruct
from torchvision import models, transforms


APP_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("IMAGE_STORAGE_DIR", APP_ROOT / "data" / "images"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "items")
API_KEY = os.getenv("IMAGE_SEARCH_API_KEY")

MODEL_DEVICE = "cpu"
MODEL_WEIGHTS = models.EfficientNet_B0_Weights.DEFAULT
MODEL = models.efficientnet_b0(weights=MODEL_WEIGHTS)
MODEL.classifier = torch.nn.Identity()
MODEL.eval()
MODEL.to(MODEL_DEVICE)

PREPROCESS = transforms.Compose(
	[
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]
)


def get_qdrant_client() -> QdrantClient:
	return QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient) -> None:
	if client.collection_exists(COLLECTION_NAME):
		return
	client.recreate_collection(
		COLLECTION_NAME,
		vectors_config=VectorParams(size=1280, distance=Distance.COSINE),
	)


def verify_api_key(header_key: Optional[str] = Header(None, alias="Image-Search-Api-Key")) -> None:
	if API_KEY is None:
		return
	if header_key != API_KEY:
		raise HTTPException(status_code=401, detail="Invalid API key")


def load_image_from_bytes(data: bytes) -> Image.Image:
	try:
		img = Image.open(io.BytesIO(data)).convert("RGB")
		return img
	except Exception as exc:  # pylint: disable=broad-except
		raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def fetch_image(url: str) -> Image.Image:
	try:
		resp = requests.get(url, timeout=10)
		resp.raise_for_status()
	except Exception as exc:  # pylint: disable=broad-except
		raise HTTPException(status_code=400, detail=f"Could not fetch image: {exc}") from exc
	return load_image_from_bytes(resp.content)


def embed_image(image: Image.Image) -> List[float]:
	tensor = PREPROCESS(image).unsqueeze(0).to(MODEL_DEVICE)
	with torch.no_grad():
		vec = MODEL(tensor).squeeze().cpu().numpy().astype(float)
	return vec.tolist()


def save_image_locally(image: Image.Image, item_id: str) -> str:
	fname = f"{item_id}_{uuid.uuid4().hex}.jpg"
	fpath = STORAGE_DIR / fname
	image.save(fpath, format="JPEG", quality=90)
	return str(fpath)


app = FastAPI(title="Image Reverse Search", version="0.2.0")


@app.on_event("startup")
def startup_event() -> None:
	client = get_qdrant_client()
	ensure_collection(client)


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/ingest")
def ingest(
	_: Optional[str] = Depends(verify_api_key),
	item_id: str = Form(...),
	item_name: Optional[str] = Form(None),
	item_code: Optional[str] = Form(None),
	image_url: Optional[str] = Form(None),
	file: Optional[UploadFile] = File(None),
) -> dict:
	if not file and not image_url:
		raise HTTPException(status_code=400, detail="Provide file or image_url")

	client = get_qdrant_client()
	ensure_collection(client)

	# Duplicate guard — check before loading/saving the image
	existing, _ = client.scroll(
		collection_name=COLLECTION_NAME,
		scroll_filter=Filter(
			must=[FieldCondition(key="item_id", match=MatchValue(value=item_id))]
		),
		limit=1,
		with_payload=False,
		with_vectors=False,
	)
	if existing:
		raise HTTPException(status_code=409, detail="item_id already exists")

	if file:
		data = file.file.read()
		image = load_image_from_bytes(data)
	else:
		image = fetch_image(image_url)  # type: ignore[arg-type]

	saved_path = save_image_locally(image, item_id)
	vector = embed_image(image)

	point_id = uuid.uuid4().hex
	payload = {
		"item_id": item_id,
		"item_name": item_name,
		"item_code": item_code,
		"image_path": saved_path,
		"source_url": image_url if image_url else saved_path,
	}
	point = PointStruct(id=point_id, vector=vector, payload=payload)
	client.upsert(collection_name=COLLECTION_NAME, points=[point])

	return {
		"status": "indexed",
		"point_id": point_id,
		"image_path": saved_path,
		"source_url": image_url,
	}


@app.post("/search")
def search(
	_: Optional[str] = Depends(verify_api_key),
	top_k: int = Form(5),
	image_url: Optional[str] = Form(None),
	file: Optional[UploadFile] = File(None),
) -> dict:
	if not file and not image_url:
		raise HTTPException(status_code=400, detail="Provide file or image_url")

	if file:
		data = file.file.read()
		image = load_image_from_bytes(data)
	else:
		image = fetch_image(image_url)  # type: ignore[arg-type]

	vector = embed_image(image)
	client = get_qdrant_client()
	ensure_collection(client)

	search_result = client.search(
		collection_name=COLLECTION_NAME,
		query_vector=vector,
		limit=top_k,
		with_payload=True,
	)

	matches = []
	for point in search_result:
		payload = point.payload or {}
		matches.append(
			{
				"score": point.score,
				"item_id": payload.get("item_id"),
				"item_name": payload.get("item_name"),
				"item_code": payload.get("item_code"),
				"image_path": payload.get("image_path"),
				"source_url": payload.get("source_url"),
			}
		)

	return {"results": matches}


@app.delete("/items/{item_id}")
def delete_item(
	item_id: str,
	_: Optional[str] = Depends(verify_api_key),
) -> dict:
	client = get_qdrant_client()
	ensure_collection(client)

	points, _ = client.scroll(
		collection_name=COLLECTION_NAME,
		scroll_filter=Filter(
			must=[FieldCondition(key="item_id", match=MatchValue(value=item_id))]
		),
		limit=100,
		with_payload=True,
		with_vectors=False,
	)

	if not points:
		raise HTTPException(status_code=404, detail="item_id not found")

	point_ids = [p.id for p in points]
	image_paths = [p.payload.get("image_path") for p in points if p.payload]

	client.delete(
		collection_name=COLLECTION_NAME,
		points_selector=point_ids,
	)

	deleted_files = []
	for path_str in image_paths:
		if path_str:
			path = Path(path_str)
			if path.exists():
				path.unlink()
				deleted_files.append(str(path))

	return {
		"status": "deleted",
		"item_id": item_id,
		"points_deleted": len(point_ids),
		"files_deleted": deleted_files,
	}


@app.delete("/items")
def delete_all_items(
	_: Optional[str] = Depends(verify_api_key),
) -> dict:
	client = get_qdrant_client()

	client.delete_collection(COLLECTION_NAME)
	client.recreate_collection(
		COLLECTION_NAME,
		vectors_config=VectorParams(size=1280, distance=Distance.COSINE),
	)

	deleted_files = 0
	for f in STORAGE_DIR.glob("*"):
		if f.is_file():
			f.unlink()
			deleted_files += 1

	return {
		"status": "all_deleted",
		"files_deleted": deleted_files,
	}

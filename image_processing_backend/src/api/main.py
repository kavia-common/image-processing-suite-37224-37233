from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Local imports (same package)
# We keep processing pure functions in this module for clarity and testability.
# Storage utils manage file system and metadata JSON store.

# PUBLIC_INTERFACE
class ResizeOp(BaseModel):
    """Resize operation with explicit width and height."""
    width: int = Field(..., description="Target width in pixels", gt=0)
    height: int = Field(..., description="Target height in pixels", gt=0)


# PUBLIC_INTERFACE
class CropOp(BaseModel):
    """Crop operation specifying top-left and dimensions."""
    x: int = Field(..., description="Top-left X coordinate", ge=0)
    y: int = Field(..., description="Top-left Y coordinate", ge=0)
    width: int = Field(..., description="Crop width", gt=0)
    height: int = Field(..., description="Crop height", gt=0)


# PUBLIC_INTERFACE
class BlurOp(BaseModel):
    """Gaussian blur operation settings."""
    radius: float = Field(..., description="Blur radius", gt=0)


# PUBLIC_INTERFACE
class BrightnessOp(BaseModel):
    """Brightness adjustment factor (1.0 keeps original)."""
    factor: float = Field(..., description="Brightness factor (1 keeps original)", gt=0)


# PUBLIC_INTERFACE
class ContrastOp(BaseModel):
    """Contrast adjustment factor (1.0 keeps original)."""
    factor: float = Field(..., description="Contrast factor (1 keeps original)", gt=0)


# PUBLIC_INTERFACE
class Operations(BaseModel):
    """Container for supported operations. Multiple can be combined."""
    resize: Optional[ResizeOp] = Field(None, description="Resize operation")
    crop: Optional[CropOp] = Field(None, description="Crop operation")
    grayscale: Optional[bool] = Field(False, description="Convert to grayscale")
    blur: Optional[BlurOp] = Field(None, description="Gaussian blur")
    brightness: Optional[BrightnessOp] = Field(None, description="Brightness adjustment")
    contrast: Optional[ContrastOp] = Field(None, description="Contrast adjustment")


# PUBLIC_INTERFACE
class UploadResponse(BaseModel):
    """Response after uploading an image."""
    image_id: str = Field(..., description="Stable ID for the uploaded image")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the uploaded file")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: str = Field(..., description="ISO timestamp when uploaded")


# PUBLIC_INTERFACE
class ProcessRequest(BaseModel):
    """Request to process an existing image using one or more operations."""
    image_id: str = Field(..., description="ID of the original image to process")
    operations: Operations = Field(..., description="Operations to apply")


# PUBLIC_INTERFACE
class VariantInfo(BaseModel):
    """Metadata for a processed variant."""
    variant_id: str = Field(..., description="ID of the processed variant")
    image_id: str = Field(..., description="Original image ID")
    filename: str = Field(..., description="Stored filename for variant")
    operations: Dict[str, Any] = Field(..., description="Operations applied")
    size_bytes: int = Field(..., description="Variant file size in bytes")
    created_at: str = Field(..., description="ISO timestamp when processed")


# PUBLIC_INTERFACE
class ImageInfo(BaseModel):
    """Metadata for an original image and its variants."""
    image_id: str = Field(..., description="Original image ID")
    filename: str = Field(..., description="Stored filename for original")
    content_type: str = Field(..., description="MIME type")
    size_bytes: int = Field(..., description="Size of original in bytes")
    created_at: str = Field(..., description="ISO timestamp when uploaded")
    variants: List[VariantInfo] = Field(default_factory=list, description="Processed variants")


# Storage and processing helpers
STORAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "storage")
ORIGINALS_DIR = os.path.abspath(os.path.join(STORAGE_ROOT, "originals"))
PROCESSED_DIR = os.path.abspath(os.path.join(STORAGE_ROOT, "processed"))
META_STORE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "meta_store.json"))

# Ensure directories exist at import time
os.makedirs(ORIGINALS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(META_STORE_PATH), exist_ok=True)
if not os.path.exists(META_STORE_PATH):
    with open(META_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump({"images": {}, "variants": {}}, f, indent=2)


def _load_meta() -> Dict[str, Any]:
    with open(META_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_meta(meta: Dict[str, Any]) -> None:
    with open(META_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _infer_extension(content_type: str, fallback: str = ".bin") -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return mapping.get(content_type.lower(), fallback)


# Pure image processing functions using Pillow
from PIL import Image, ImageFilter, ImageOps, ImageEnhance  # noqa: E402


def _apply_operations(img: Image.Image, ops: Operations) -> Image.Image:
    """Pure function that applies operations to a Pillow image and returns a new image instance."""
    result = img
    # crop first if given
    if ops.crop:
        x, y, w, h = ops.crop.x, ops.crop.y, ops.crop.width, ops.crop.height
        result = result.crop((x, y, x + w, y + h))
    # resize
    if ops.resize:
        result = result.resize((ops.resize.width, ops.resize.height))
    # grayscale
    if ops.grayscale:
        result = ImageOps.grayscale(result).convert("RGB")
    # blur
    if ops.blur:
        result = result.filter(ImageFilter.GaussianBlur(radius=ops.blur.radius))
    # brightness
    if ops.brightness:
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(ops.brightness.factor)
    # contrast
    if ops.contrast:
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(ops.contrast.factor)
    return result


def _image_to_bytes(img: Image.Image, format_hint: Optional[str]) -> bytes:
    """Serialize Pillow image to bytes, using a sensible format based on hint."""
    buf = io.BytesIO()
    fmt = None
    if format_hint:
        # Normalize common extensions to Pillow format names
        ext = format_hint.lower().strip(".")
        if ext in ("jpg", "jpeg"):
            fmt = "JPEG"
        elif ext == "png":
            fmt = "PNG"
        elif ext == "gif":
            fmt = "GIF"
        elif ext == "webp":
            fmt = "WEBP"
        elif ext == "bmp":
            fmt = "BMP"
        elif ext in ("tif", "tiff"):
            fmt = "TIFF"
    # Default to PNG for safety if unknown
    img.save(buf, format=fmt or "PNG")
    return buf.getvalue()


openapi_tags = [
    {"name": "health", "description": "Service health"},
    {"name": "images", "description": "Upload, process, list, and retrieve images"},
]

app = FastAPI(
    title="Image Processing Suite API",
    description="Upload images, apply basic processing operations (resize, crop, filters), and retrieve originals or processed variants.",
    version="1.0.0",
    openapi_tags=openapi_tags,
)

# CORS for frontend at http://localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PUBLIC_INTERFACE
@app.get("/", tags=["health"], summary="Health Check", description="Basic health check endpoint.")
def health_check() -> Dict[str, str]:
    """Return a simple JSON to indicate the service is running."""
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.post("/images/upload", tags=["images"], summary="Upload an image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """Upload an image file via multipart/form-data. Stores the original and returns its metadata."""
    content_type = file.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image.")

    image_id = uuid.uuid4().hex
    ext = _infer_extension(content_type, fallback=".bin")
    stored_filename = f"{image_id}{ext}"
    stored_path = os.path.join(ORIGINALS_DIR, stored_filename)

    raw = await file.read()
    try:
        # Validate that Pillow can open it
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    with open(stored_path, "wb") as f:
        f.write(raw)

    size_bytes = os.path.getsize(stored_path)
    created_at = datetime.utcnow().isoformat() + "Z"

    meta = _load_meta()
    meta["images"][image_id] = {
        "image_id": image_id,
        "filename": stored_filename,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "created_at": created_at,
        "variants": [],
    }
    _save_meta(meta)

    return UploadResponse(
        image_id=image_id,
        filename=stored_filename,
        content_type=content_type,
        size_bytes=size_bytes,
        created_at=created_at,
    )


# PUBLIC_INTERFACE
@app.post(
    "/images/process",
    tags=["images"],
    summary="Process an image",
    response_model=VariantInfo,
)
async def process_image(req: ProcessRequest) -> VariantInfo:
    """Apply operations to a stored original image and save the processed result as a variant."""
    meta = _load_meta()
    if req.image_id not in meta["images"]:
        raise HTTPException(status_code=404, detail="Original image not found.")

    original = meta["images"][req.image_id]
    original_path = os.path.join(ORIGINALS_DIR, original["filename"])
    if not os.path.exists(original_path):
        raise HTTPException(status_code=410, detail="Original image file missing.")

    # Load and process
    try:
        img = Image.open(original_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to open original image.")

    processed = _apply_operations(img, req.operations)

    variant_id = uuid.uuid4().hex
    # Store variant using PNG to ensure compatibility unless original has a defined extension
    ext = os.path.splitext(original["filename"])[1] or ".png"
    variant_filename = f"{variant_id}{ext}"
    variant_path = os.path.join(PROCESSED_DIR, variant_filename)

    out_bytes = _image_to_bytes(processed, ext)
    with open(variant_path, "wb") as f:
        f.write(out_bytes)

    size_bytes = os.path.getsize(variant_path)
    created_at = datetime.utcnow().isoformat() + "Z"

    variant_meta = {
        "variant_id": variant_id,
        "image_id": req.image_id,
        "filename": variant_filename,
        "operations": json.loads(req.operations.model_dump_json()),
        "size_bytes": size_bytes,
        "created_at": created_at,
    }
    meta["variants"][variant_id] = variant_meta
    # Also append to the image's variants list (store a lightweight copy)
    original["variants"].append(variant_meta)
    _save_meta(meta)

    return VariantInfo(**variant_meta)


# PUBLIC_INTERFACE
@app.get(
    "/images",
    tags=["images"],
    summary="List images",
    response_model=List[ImageInfo],
)
def list_images(limit: int = Query(50, ge=1, le=200)) -> List[ImageInfo]:
    """List recent images with their variants. Results are sorted by created_at desc."""
    meta = _load_meta()
    images = list(meta["images"].values())
    images.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    images = images[:limit]
    return [ImageInfo(**img) for img in images]


# PUBLIC_INTERFACE
@app.get(
    "/images/{image_id}",
    tags=["images"],
    summary="Get original image bytes",
    responses={
        200: {"content": {"image/*": {}}},
        404: {"description": "Not Found"},
    },
)
def get_original_image(image_id: str) -> Response:
    """Serve the original image bytes for the given image_id."""
    meta = _load_meta()
    if image_id not in meta["images"]:
        raise HTTPException(status_code=404, detail="Image not found")
    item = meta["images"][image_id]
    path = os.path.join(ORIGINALS_DIR, item["filename"])
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="Image file missing")
    with open(path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=item.get("content_type", "application/octet-stream"))


# PUBLIC_INTERFACE
@app.get(
    "/images/{image_id}/variants",
    tags=["images"],
    summary="List variants for an image",
    response_model=List[VariantInfo],
)
def list_variants(image_id: str) -> List[VariantInfo]:
    """List processed variants for the given image."""
    meta = _load_meta()
    if image_id not in meta["images"]:
        raise HTTPException(status_code=404, detail="Image not found")
    variants = meta["images"][image_id].get("variants", [])
    return [VariantInfo(**v) for v in variants]


# PUBLIC_INTERFACE
@app.get(
    "/images/processed/{variant_id}",
    tags=["images"],
    summary="Get processed variant bytes",
    responses={
        200: {"content": {"image/*": {}}},
        404: {"description": "Not Found"},
    },
)
def get_processed_image(variant_id: str) -> StreamingResponse:
    """Serve processed image bytes for the given variant_id."""
    meta = _load_meta()
    if variant_id not in meta["variants"]:
        raise HTTPException(status_code=404, detail="Variant not found")
    item = meta["variants"][variant_id]
    path = os.path.join(PROCESSED_DIR, item["filename"])
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="Variant file missing")
    # Attempt to infer media type from extension
    ext = os.path.splitext(item["filename"])[1].lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    media_type = media_map.get(ext, "application/octet-stream")

    def iterator():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(iterator(), media_type=media_type)

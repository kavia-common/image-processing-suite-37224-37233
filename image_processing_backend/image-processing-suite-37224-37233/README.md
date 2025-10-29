# image-processing-suite-37224-37233

This repository hosts the Image Processing Suite.

Backend (FastAPI):
- Upload images, process (resize, crop, grayscale, blur, brightness, contrast), retrieve originals and processed variants.
- Storage on disk under `storage/originals` and `storage/processed`.
- Metadata stored in `image_processing_backend/src/app/meta_store.json`.

Run (example, actual command provided by preview system):
- The preview system starts the FastAPI server automatically.

API Examples (assuming backend at http://localhost:3001):

1) Health
curl http://localhost:3001/

2) Upload
curl -F "file=@/path/to/pic.jpg" http://localhost:3001/images/upload

3) Process
curl -X POST http://localhost:3001/images/process \
 -H "Content-Type: application/json" \
 -d '{
  "image_id": "<IMAGE_ID_FROM_UPLOAD>",
  "operations": {
    "resize": {"width": 400, "height": 300},
    "grayscale": true,
    "blur": {"radius": 1.5},
    "brightness": {"factor": 1.2},
    "contrast": {"factor": 0.9}
  }
}'

4) List images
curl http://localhost:3001/images

5) Get original image
curl -L http://localhost:3001/images/<IMAGE_ID> --output original.jpg

6) List variants
curl http://localhost:3001/images/<IMAGE_ID>/variants

7) Get processed variant
curl -L http://localhost:3001/images/processed/<VARIANT_ID> --output processed.png

CORS:
- Enabled for http://localhost:3000 for the React frontend (retro themed).

# Image Reverse Search

FastAPI + EfficientNet-B0 + Qdrant for local image similarity search. Images are saved to the local filesystem (`./data/images`). No S3 required.

## Quick start

```bash
cd reverse-image-search
# build and start qdrant + api
docker compose up --build
# API available at http://localhost:8020
```

Optional env vars (set in `docker-compose.yml` or shell):
- `IMAGE_SEARCH_API_KEY`: shared secret for ERPNext → API calls (omit for no auth)
- `QDRANT_URL` (default `http://qdrant:6333`)
- `QDRANT_COLLECTION` (default `items`)
- `IMAGE_STORAGE_DIR` (default `./data/images`)

## Endpoints

- `GET /health` → `{status: "ok"}`
- `POST /ingest` (multipart): `item_id` (required), `item_name?`, `item_code?`, `file` **or** `image_url`
  - Returns `409` if `item_id` already exists
- `POST /search` (multipart): `top_k` (default 5), `file` **or** `image_url`
- `DELETE /items/{item_id}` — deletes all Qdrant points and local image files for the given item_id (404 if not found)
- `DELETE /items` — deletes ALL points from the collection and all files in `IMAGE_STORAGE_DIR`

All endpoints require the `Image-Search-Api-Key` header when `IMAGE_SEARCH_API_KEY` is set.

## Notes

- Uses EfficientNet-B0 (ImageNet weights) on CPU; vector dim = 1280; cosine distance.
- Qdrant collection is created on startup if missing.
- For ERPNext integration, call these endpoints from `general_app_ascend` hooks/client scripts and pass the file stream or the ERPNext file URL.

## Migrating from MobileNetV3-Small

If you have an existing Qdrant collection with 576-dim vectors, drop it before restarting — the startup event will recreate it at 1280-dim:

```bash
curl -X DELETE http://localhost:6333/collections/items
docker compose up --build
```

---

## AWS Deployment (EC2 t4.medium)

### 1. Launch instance

- **Instance type:** t4g.medium (Graviton2, arm64) or t3.medium (x86_64)
- **AMI:** Ubuntu 22.04 LTS or Amazon Linux 2023
- **Storage:** root volume 20 GB + attach a separate EBS gp3 volume (20–50 GB) for images and Qdrant data

### 2. Mount the EBS volume

```bash
# Find the device name (usually /dev/nvme1n1 or /dev/xvdb)
lsblk

sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /data
echo '/dev/nvme1n1 /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a
sudo mkdir -p /data/images
```

### 3. Install Docker + Compose

**Ubuntu 22.04:**
```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
# log out and back in to pick up the group
```

**Amazon Linux 2023:**
```bash
sudo yum install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
# log out and back in
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 4. Deploy the service

```bash
# Copy project to instance (from local machine)
scp -r ./reverse-image-search ec2-user@<EC2-IP>:/data/reverse-image-search

# On the instance
cd /data/reverse-image-search

# Edit docker-compose.yml:
#   - Set IMAGE_SEARCH_API_KEY to a strong secret
#   - Set IMAGE_STORAGE_DIR=/data/images

docker compose up --build -d
```

Update `docker-compose.yml` volumes to point to the mounted EBS:
```yaml
volumes:
  - /data/images:/app/data/images
```

And add the Qdrant storage bind mount for EBS persistence:
```yaml
qdrant:
  volumes:
    - /data/qdrant:/qdrant/storage
```

### 5. EC2 Security Group rules

| Type | Port | Source |
|---|---|---|
| SSH | 22 | Your IP only |
| Custom TCP | 8020 | Your app server IP / VPC CIDR |

Do **not** expose port 6333 (Qdrant) to the internet.

### 6. Verify

```bash
curl http://<EC2-public-IP>:8020/health
# → {"status":"ok"}
```

### Notes

- Both services use `restart: unless-stopped` — they auto-start on instance reboot.
- EfficientNet-B0 inference on CPU: ~150–300 ms per image on t3.medium. Acceptable for ingest; search is fast via Qdrant HNSW.
- Memory footprint: model ~21 MB, Qdrant with thousands of 1280-dim vectors stays well under 1 GB — comfortable within 4 GB RAM.
- **Qdrant data backup:** use the Qdrant snapshot API (`POST /collections/items/snapshots`) before replacing the instance, or back up `/data/qdrant` directly.

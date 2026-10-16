import urllib.request
import json

tests = [
    ("backend/uploads/cyste 3.jpg", "cyste 3.jpg"),
    ("backend/uploads/Tumor.jpg", "Tumor.jpg"),
    ("backend/uploads/stone.webp", "stone.webp"),
    ("backend/uploads/Normal.jpg", "Normal.jpg")
]

boundary = "BOUNDARY12345"
for path, fname in tests:
    with open(path, "rb") as f:
        img = f.read()
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n",
        img,
        f"\r\n--{boundary}--\r\n".encode()
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        "https://nephrovision-ui.onrender.com/api/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as res:
            data = json.loads(res.read().decode())
            print(f"[{fname}] Result: {data.get('classification')} | is_cyst={data.get('is_cyst')} | is_tumor={data.get('is_tumor')} | is_stone={data.get('is_stone')} | boxes={len(data.get('bounding_boxes', []))}")
    except Exception as e:
        print(f"[{fname}] Error: {e}")

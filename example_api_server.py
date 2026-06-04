from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageOps
import io


app = Flask(__name__)


@app.post("/edit")
def edit():
    image_file = request.files["image"]
    mask_file = request.files["mask"]
    image = Image.open(image_file.stream).convert("RGB")
    mask = Image.open(mask_file.stream).convert("L")

    # Demo only: draw a pink overlay on the masked region so paste-back is visible.
    overlay = Image.new("RGB", image.size, (255, 45, 141))
    edited = Image.composite(overlay, image, ImageOps.autocontrast(mask))

    buffer = io.BytesIO()
    edited.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787)

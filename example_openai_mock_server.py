import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from cgi import FieldStorage

from PIL import Image, ImageOps


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/images/edits":
            self.send_error(404)
            return

        form = FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )
        image = Image.open(form["image"].file).convert("RGB")
        mask = Image.open(form["mask"].file).convert("RGBA")
        alpha = mask.getchannel("A")
        edit_mask = ImageOps.invert(alpha)
        overlay = Image.new("RGB", image.size, (255, 45, 141))
        edited = Image.composite(overlay, image, edit_mask)

        buffer = io.BytesIO()
        edited.save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        body = json.dumps({"data": [{"b64_json": payload}]}).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8788), Handler).serve_forever()

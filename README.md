# ComfyUI-Mask-External-Edit

ComfyUI custom node for local masked image editing.

The node accepts an original image and a user-painted mask, crops the masked area with context padding, sends the crop and crop mask to an image-editing provider such as OpenAI, then blends the edited crop back into the original image.

This is designed for product features such as:

- Fix hands
- Enhance face
- Change expression
- Repair local generation bugs
- General masked image editing

## Node

After installation, search this node in ComfyUI:

```text
Mask External Edit
```

Category:

```text
Mask External Edit / Enhance
```

## Inputs

| Input | Description |
| --- | --- |
| `image` | Original ComfyUI image |
| `mask` | User-painted mask |
| `provider` | `openai`, `custom_http`, or `debug_echo` |
| `task` | `general_fix`, `fix_hands`, `enhance_face`, `change_expression` |
| `prompt` | Edit instruction sent to the external service |
| `api_key` | Optional API key entered directly in the node |
| `api_key_env` | Optional environment variable name for the API key |
| `api_endpoint` | Optional OpenAI image edits endpoint when `provider=openai`; leave empty for official OpenAI |
| `padding` | Context pixels added around the mask crop |
| `mask_grow` | Expands the mask before crop/paste |
| `feather` | Softens pasted edge |
| `crop_max_size` | Max crop size sent to the external API |
| `threshold` | Mask threshold for finding the painted area |
| `timeout_seconds` | HTTP request timeout |
| `blend_mode` | `normal` or `color_match` |
| `openai_model` | OpenAI image edit model name, for example `gpt-image-2` |
| `mask_mode` | `auto`, `white_edits`, or `black_edits` |
| `on_error` | `raise` to show API errors, or `return_original` to silently return the original image |

## Outputs

| Output | Description |
| --- | --- |
| `image` | Final image after paste-back |
| `debug_crop` | Original crop sent for editing |
| `edited_crop` | External API result crop |
| `used_mask` | Final paste mask |
| `status` | Request status or failure reason |

## Install

Manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wuhu290/ComfyUI-Mask-External-Edit.git
cd ComfyUI-Mask-External-Edit
pip install -r requirements.txt
```

Restart ComfyUI.

Then right click in the canvas and search:

```text
Mask External Edit
```

If you use ComfyUI Manager and the repository is not listed yet:

```text
Manager -> Install via Git URL -> paste the GitHub repository URL
```

To make it searchable in ComfyUI Manager for everyone, publish the repository on GitHub and submit it to the ComfyUI Registry or Manager list. Before publishing, update these fields in `pyproject.toml`:

```toml
[project.urls]
Repository = "https://github.com/wuhu290/ComfyUI-Mask-External-Edit"

[tool.comfy]
PublisherId = "wuhu290"
```

Official references:

- ComfyUI custom node install: https://docs.comfy.org/installation/install_custom_node
- ComfyUI Registry metadata: https://docs.comfy.org/registry/specifications
- Publishing nodes: https://docs.comfy.org/registry/publishing

## OpenAI usage

For direct OpenAI image editing inside ComfyUI:

```text
provider: openai
api_key: your OpenAI API key
openai_model: gpt-image-2
api_endpoint: leave empty
```

The node calls:

```text
POST https://api.openai.com/v1/images/edits
```

If you use an OpenAI-compatible gateway or proxy, fill `api_endpoint` with either its base URL or full image edit URL:

```text
https://your-gateway.example.com/v1
https://your-gateway.example.com/v1/images/edits
```

It sends the cropped image and a generated PNG mask with an alpha channel. In ComfyUI, the area you paint in the mask is treated as the area to edit.

OpenAI notes that GPT Image masks are prompt-guided and may not follow the mask shape with pixel-perfect precision. See the official image editing guide and image edits API reference:

- https://platform.openai.com/docs/guides/image-generation
- https://platform.openai.com/docs/api-reference/images/createEdit

## Custom HTTP API contract

The `custom_http` provider sends a `multipart/form-data` POST request:

```text
POST {api_endpoint}

files:
  image: crop.png
  mask: mask.png

fields:
  prompt: string
  task: general_fix | fix_hands | enhance_face | change_expression
```

If `api_key` is filled, the node sends:

```text
Authorization: Bearer {api_key}
```

If `api_key` is empty and `api_key_env` is set and the environment variable exists, the node sends:

```text
Authorization: Bearer {API_KEY}
```

Example:

```bash
set MASK_EXTERNAL_EDIT_API_KEY=your_key_here
```

On Windows PowerShell:

```powershell
$env:MASK_EXTERNAL_EDIT_API_KEY="your_key_here"
```

Expected response can be any one of:

1. Direct image response:

```text
Content-Type: image/png
```

2. JSON with base64:

```json
{
  "image_base64": "..."
}
```

3. JSON with URL:

```json
{
  "url": "https://example.com/edited.png"
}
```

4. JSON with image list:

```json
{
  "images": [
    {
      "image_base64": "..."
    }
  ]
}
```

## Recommended settings

| Task | Padding | Mask Grow | Feather |
| --- | ---: | ---: | ---: |
| Fix hands | 160-256 | 8-20 | 20-48 |
| Enhance face | 128-192 | 6-16 | 16-36 |
| Change expression | 160-256 | 4-12 | 18-40 |
| General local bug fix | 96-192 | 8-20 | 20-48 |

Use `debug_echo` first to verify crop, mask, and paste-back before connecting a real API.

For debugging, keep:

```text
on_error: raise
```

This makes API/model/key/mask problems visible in ComfyUI instead of returning an unchanged original image.

For masks created by ComfyUI MaskEditor / clipspace, keep:

```text
mask_mode: auto
```

If the edited area is inverted, switch manually:

```text
white_edits: white mask pixels are edited
black_edits: black/dark mask pixels are edited
```

## Notes

- This node does not bypass external provider safety policies.
- If the external API fails, the node returns the original image and writes the reason to `status`.
- Direct `api_key` input is convenient for local testing, but exported workflows and screenshots may expose it. Use `api_key_env` for shared or production workflows.
- `debug_echo` is included only for checking mask crop and paste-back behavior before connecting a real provider.

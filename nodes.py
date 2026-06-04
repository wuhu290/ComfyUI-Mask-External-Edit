import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageFilter, ImageOps


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    array = image.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        return Image.fromarray(array, "RGBA").convert("RGB")
    return Image.fromarray(array, "RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _mask_to_pil(mask: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    if mask.ndim == 3:
        mask = mask[0]
    array = mask.detach().cpu().numpy()
    array = np.squeeze(array)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _prepare_edit_mask(mask: Image.Image, mask_mode: str) -> Image.Image:
    normalized = ImageOps.autocontrast(mask.convert("L"))
    if mask_mode == "white_edits":
        return normalized
    if mask_mode == "black_edits":
        return ImageOps.invert(normalized)

    values = np.asarray(normalized).astype(np.float32)
    # ComfyUI MaskEditor/clipspace alpha masks often store painted areas as darker alpha
    # over an otherwise opaque image. In that case the bright area dominates, so invert.
    if float(values.mean()) > 127.0:
        return ImageOps.invert(normalized)
    return normalized


def _pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _safe_odd_size(radius: int) -> int:
    return max(3, int(radius) * 2 + 1)


def _mask_bbox(mask: Image.Image, threshold: int) -> Tuple[int, int, int, int] | None:
    mask = mask.convert("L")
    binary = mask.point(lambda value: 255 if value >= threshold else 0)
    return binary.getbbox()


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    padding: int,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _resize_for_limit(
    image: Image.Image,
    mask: Image.Image,
    max_size: int,
) -> Tuple[Image.Image, Image.Image, float]:
    if max_size <= 0:
        return image, mask, 1.0
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image, mask, 1.0
    scale = max_size / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return (
        image.resize(new_size, Image.Resampling.LANCZOS),
        mask.resize(new_size, Image.Resampling.LANCZOS),
        scale,
    )


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _openai_mask_from_user_mask(mask: Image.Image) -> Image.Image:
    edit_mask = ImageOps.autocontrast(mask.convert("L"))
    alpha = ImageOps.invert(edit_mask)
    rgba = Image.new("RGBA", mask.size, (255, 255, 255, 255))
    rgba.putalpha(alpha)
    return rgba


def _decode_response_image(response: requests.Response) -> Image.Image:
    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith("image/"):
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    data = response.json()
    if "data" in data and data["data"]:
        item = data["data"][0]
        if isinstance(item, dict):
            payload = item.get("b64_json") or item.get("image_base64") or item.get("base64")
            if payload:
                if "," in payload and payload.strip().startswith("data:"):
                    payload = payload.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
            url = item.get("url")
            if url:
                remote = requests.get(url, timeout=60)
                remote.raise_for_status()
                return Image.open(io.BytesIO(remote.content)).convert("RGB")
    if "image_base64" in data:
        payload = data["image_base64"]
        if "," in payload and payload.strip().startswith("data:"):
            payload = payload.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if "images" in data and data["images"]:
        payload = data["images"][0]
        if isinstance(payload, dict):
            payload = payload.get("image_base64") or payload.get("base64") or payload.get("url")
        if isinstance(payload, str) and payload.startswith("http"):
            remote = requests.get(payload, timeout=60)
            remote.raise_for_status()
            return Image.open(io.BytesIO(remote.content)).convert("RGB")
        if isinstance(payload, str):
            if "," in payload and payload.strip().startswith("data:"):
                payload = payload.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if "url" in data:
        remote = requests.get(data["url"], timeout=60)
        remote.raise_for_status()
        return Image.open(io.BytesIO(remote.content)).convert("RGB")
    raise ValueError("API response did not contain an image, image_base64, images, or url field.")


def _raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:1000] if response.text else ""
        raise requests.HTTPError(f"{exc}; response body: {body}", response=response) from exc


def _resolve_api_key(api_key: str, api_key_env: str) -> str:
    resolved_api_key = api_key.strip()
    if not resolved_api_key and api_key_env.strip():
        resolved_api_key = os.getenv(api_key_env.strip(), "")
    return resolved_api_key


def _call_custom_http(
    endpoint: str,
    api_key: str,
    api_key_env: str,
    prompt: str,
    crop: Image.Image,
    crop_mask: Image.Image,
    task: str,
    timeout: int,
) -> Image.Image:
    if not endpoint.strip():
        raise ValueError("api_endpoint is required for custom_http provider.")

    headers: Dict[str, str] = {}
    resolved_api_key = _resolve_api_key(api_key, api_key_env)
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    files = {
        "image": ("crop.png", _image_to_png_bytes(crop), "image/png"),
        "mask": ("mask.png", _image_to_png_bytes(crop_mask), "image/png"),
    }
    data = {
        "prompt": prompt,
        "task": task,
    }
    response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=timeout)
    _raise_for_status_with_body(response)
    return _decode_response_image(response)


def _call_openai_edit(
    endpoint: str,
    api_key: str,
    api_key_env: str,
    prompt: str,
    crop: Image.Image,
    crop_mask: Image.Image,
    task: str,
    model: str,
    timeout: int,
) -> Image.Image:
    resolved_api_key = _resolve_api_key(api_key, api_key_env)
    if not resolved_api_key:
        raise ValueError("OpenAI API key is required. Fill api_key or api_key_env.")

    instruction = (
        f"Task: {task}. {prompt}\n"
        "Only edit the masked transparent area. Preserve the unmasked area, character identity, pose, "
        "style, lighting, colors, and background as much as possible."
    )
    openai_mask = _openai_mask_from_user_mask(crop_mask)
    files = {
        "image": ("image.png", _image_to_png_bytes(crop.convert("RGB")), "image/png"),
        "mask": ("mask.png", _image_to_png_bytes(openai_mask), "image/png"),
    }
    data = {
        "model": model,
        "prompt": instruction,
        "n": "1",
    }
    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
    }
    edit_endpoint = endpoint.strip() or "https://api.openai.com/v1/images/edits"
    response = requests.post(
        edit_endpoint,
        headers=headers,
        files=files,
        data=data,
        timeout=timeout,
    )
    _raise_for_status_with_body(response)
    return _decode_response_image(response)


def _images_are_identical(left: Image.Image, right: Image.Image) -> bool:
    if left.size != right.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)
    left_arr = np.asarray(left.convert("RGB"))
    right_arr = np.asarray(right.convert("RGB"))
    return bool(np.array_equal(left_arr, right_arr))


@dataclass
class PasteResult:
    image: Image.Image
    mask: Image.Image


def _paste_back(
    original: Image.Image,
    edited_crop: Image.Image,
    mask: Image.Image,
    bbox: Tuple[int, int, int, int],
    feather: int,
    blend_mode: str,
) -> PasteResult:
    crop_width = bbox[2] - bbox[0]
    crop_height = bbox[3] - bbox[1]
    edited_crop = edited_crop.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    crop_mask = mask.crop(bbox).resize((crop_width, crop_height), Image.Resampling.LANCZOS)

    if feather > 0:
        crop_mask = crop_mask.filter(ImageFilter.GaussianBlur(radius=feather))

    original_crop = original.crop(bbox)
    if blend_mode == "color_match":
        edited_crop = _match_luminance(edited_crop, original_crop)

    result = original.copy()
    result.paste(edited_crop, (bbox[0], bbox[1]), crop_mask)
    full_mask = Image.new("L", original.size, 0)
    full_mask.paste(crop_mask, (bbox[0], bbox[1]))
    return PasteResult(result, full_mask)


def _match_luminance(source: Image.Image, target: Image.Image) -> Image.Image:
    source_rgb = np.asarray(source.convert("RGB")).astype(np.float32)
    target_rgb = np.asarray(target.convert("RGB")).astype(np.float32)
    source_mean = source_rgb.mean(axis=(0, 1), keepdims=True)
    source_std = source_rgb.std(axis=(0, 1), keepdims=True) + 1e-6
    target_mean = target_rgb.mean(axis=(0, 1), keepdims=True)
    target_std = target_rgb.std(axis=(0, 1), keepdims=True) + 1e-6
    matched = (source_rgb - source_mean) / source_std * target_std + target_mean
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    return Image.fromarray(matched, "RGB")


class MaskExternalEdit:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "provider": (["openai", "custom_http", "debug_echo"], {"default": "openai"}),
                "task": (["general_fix", "fix_hands", "enhance_face", "change_expression"], {"default": "general_fix"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Fix only the masked area. Preserve the original character, pose, style, lighting, and background.",
                }),
                "api_endpoint": ("STRING", {"default": ""}),
                "api_key_env": ("STRING", {"default": "MASK_EXTERNAL_EDIT_API_KEY"}),
                "padding": ("INT", {"default": 160, "min": 0, "max": 1024, "step": 8}),
                "mask_grow": ("INT", {"default": 12, "min": 0, "max": 256, "step": 2}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2}),
                "crop_max_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "threshold": ("INT", {"default": 16, "min": 1, "max": 255, "step": 1}),
                "timeout_seconds": ("INT", {"default": 120, "min": 10, "max": 600, "step": 10}),
                "blend_mode": (["normal", "color_match"], {"default": "color_match"}),
                "api_key": ("STRING", {"default": ""}),
                "openai_model": ("STRING", {"default": "gpt-image-2"}),
                "mask_mode": (["auto", "white_edits", "black_edits"], {"default": "auto"}),
                "on_error": (["raise", "return_original"], {"default": "raise"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "debug_crop", "edited_crop", "used_mask", "status")
    FUNCTION = "edit"
    CATEGORY = "Mask External Edit/Enhance"

    def edit(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        provider: str,
        task: str,
        prompt: str,
        api_endpoint: str,
        api_key_env: str,
        padding: int,
        mask_grow: int,
        feather: int,
        crop_max_size: int,
        threshold: int,
        timeout_seconds: int,
        blend_mode: str,
        api_key: str,
        openai_model: str,
        mask_mode: str,
        on_error: str,
    ):
        original = _tensor_to_pil(image)
        source_mask = _prepare_edit_mask(_mask_to_pil(mask, original.size), mask_mode)

        if mask_grow > 0:
            source_mask = source_mask.filter(ImageFilter.MaxFilter(_safe_odd_size(mask_grow)))

        bbox = _mask_bbox(source_mask, threshold)
        if bbox is None:
            status = "No masked area found. Returning original image."
            empty = Image.new("RGB", original.size, (0, 0, 0))
            return (
                _pil_to_tensor(original),
                _pil_to_tensor(empty),
                _pil_to_tensor(empty),
                _pil_mask_to_tensor(Image.new("L", original.size, 0)),
                status,
            )

        crop_bbox = _expand_bbox(bbox, padding, original.size)
        crop = original.crop(crop_bbox)
        crop_mask = source_mask.crop(crop_bbox)
        api_crop, api_mask, scale = _resize_for_limit(crop, crop_mask, crop_max_size)

        status = "ok"
        succeeded = True
        try:
            if provider == "debug_echo":
                overlay = Image.new("RGB", api_crop.size, (255, 45, 141))
                debug_mask = ImageOps.autocontrast(api_mask.convert("L"))
                edited_crop = Image.composite(overlay, api_crop.convert("RGB"), debug_mask)
                status = "debug_echo: painted the detected mask area without calling external API."
            elif provider == "openai":
                edited_crop = _call_openai_edit(
                    api_endpoint,
                    api_key,
                    api_key_env,
                    prompt,
                    api_crop,
                    api_mask,
                    task,
                    openai_model,
                    timeout_seconds,
                )
                if _images_are_identical(api_crop, edited_crop):
                    raise ValueError("OpenAI returned an unchanged crop. Check mask, prompt, model access, or API endpoint.")
                status = f"openai: edited crop {api_crop.size[0]}x{api_crop.size[1]}, model={openai_model}, scale={scale:.3f}"
            elif provider == "custom_http":
                edited_crop = _call_custom_http(
                    api_endpoint,
                    api_key,
                    api_key_env,
                    prompt,
                    api_crop,
                    api_mask,
                    task,
                    timeout_seconds,
                )
                status = f"custom_http: edited crop {api_crop.size[0]}x{api_crop.size[1]}, scale={scale:.3f}"
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as exc:
            status = f"external edit failed: {exc}. Returning original image."
            if on_error == "raise":
                raise RuntimeError(status) from exc
            edited_crop = api_crop
            succeeded = False

        if succeeded:
            pasted = _paste_back(original, edited_crop, source_mask, crop_bbox, feather, blend_mode)
            final_image = pasted.image
            used_mask = pasted.mask
        else:
            final_image = original
            used_mask = Image.new("L", original.size, 0)

        debug_crop = crop
        if debug_crop.size != api_crop.size:
            debug_crop = debug_crop.resize(api_crop.size, Image.Resampling.LANCZOS)

        return (
            _pil_to_tensor(final_image),
            _pil_to_tensor(debug_crop),
            _pil_to_tensor(edited_crop),
            _pil_mask_to_tensor(used_mask),
            status,
        )


NODE_CLASS_MAPPINGS = {
    "MaskExternalEdit": MaskExternalEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskExternalEdit": "Mask External Edit",
}

import base64
import uuid
import io
import json
import socket
import ssl
import urllib.error
import time
import threading
import os
try:
    import winreg
except ImportError:
    winreg = None
import urllib.parse
import urllib.request
import http.client

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

_BASE_URL = "https://api.smile-ai-studio.com"

MSG_VIOLATION = "内容违规，可能涉嫌色情内容，请修改图片或提示词后重新生成"
MSG_SIZE_TOO_LARGE = "图片尺寸太大 AI无法承载 请将最长边分辨率调整到6500像素以下"
MSG_TIMEOUT = "任务超时，未链接成功，请重新生成"
MSG_GENERIC = "生成失败，请稍后重试"

_VIOLATION_KEYS = [
    "safety", "content_policy", "blocklist", "blocked", "policy",
    "violat", "moderation", "disallowed", "abuse",
    "sexual", "porn", "explicit", "inappropriate", "nudity",
    "违法", "违规", "内容", "色情", "审核", "不适宜",
]
_SIZE_KEYS = [
    "size", "resolution", "dimension", "too large", "too_large",
    "too big", "exceed", "larger than", "maximum", "limit",
    "像素", "尺寸", "分辨率", "6500", "超限", "过大",
]

RESOLUTION_CHOICES = ["1K", "2K", "4K"]
RESOLUTION_CHOICES_NO_1K = ["2K", "4K"]
ASPECT_CHOICES = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4",
                  "3:2", "2:3", "5:4", "4:5", "21:9", "9:21",
                  "18:9", "16:10"]
BANANA_ASPECT_CHOICES = ["1:1", "16:9", "9:16", "4:3", "3:4",
                         "3:2", "2:3", "5:4", "4:5", "21:9"]
BANANA_ASPECT_CHOICES_EDIT = ["auto"] + BANANA_ASPECT_CHOICES

_SERIES_MODEL = {
    "banana_v2": {
        "1K": "gemini-3.1-flash-image-preview-1k",
        "2K": "gemini-3.1-flash-image-preview-2k",
        "4K": "gemini-3.1-flash-image-preview-4k",
    },
    "banana_pro": {
        "1K": "gemini-3-pro-image-preview-1k",
        "2K": "gemini-3-pro-image-preview-2k",
        "4K": "gemini-3-pro-image-preview-4k",
    },
    "banana_v21": {
        "4K": "gemini-nano-banana-2.1-4k",
    },
}

_BR_SIZES = {
    "1K": {"auto": "1024x1024", "1:1": "1024x1024", "16:9": "1536x864",
            "9:16": "864x1536", "4:3": "1360x1024", "3:4": "1024x1360",
            "3:2": "1536x1024", "2:3": "1024x1536", "5:4": "1280x1024",
            "4:5": "1024x1280", "21:9": "1536x672", "9:21": "672x1536",
            "18:9": "1536x768", "16:10": "1536x960"},
    "2K": {"auto": "2048x2048", "1:1": "2048x2048", "16:9": "2048x1152",
            "9:16": "1152x2048", "4:3": "2048x1536", "3:4": "1536x2048",
            "3:2": "2048x1360", "2:3": "1360x2048", "5:4": "2048x1632",
            "4:5": "1632x2048", "21:9": "2048x880", "9:21": "880x2048",
            "18:9": "2048x1024", "16:10": "2048x1280"},
    "4K": {"auto": "2880x2880", "1:1": "2880x2880", "16:9": "3840x2160",
            "9:16": "2160x3840", "4:3": "3312x2496", "3:4": "2496x3312",
            "3:2": "3520x2352", "2:3": "2352x3520", "5:4": "3216x2576",
            "4:5": "2576x3216", "21:9": "3840x1648", "9:21": "1648x3840",
            "18:9": "3840x1920", "16:10": "3648x2272"},
}


CATEGORY = "木木"

_IMAGE_KEYS = ["image_%d" % i for i in range(1, 11)]


def _resolve_model(series, resolution):
    table = _SERIES_MODEL[series]
    return table.get(resolution) or table.get("1K") or next(iter(table.values()))


def _extract_api_message(raw_bytes, fallback_text):
    try:
        err = json.loads(fallback_text)
        return err.get("error", {}).get("message", fallback_text)
    except Exception:
        return fallback_text


def _categorize_text(raw_text):
    t = (raw_text or "").lower()
    if any(k in t for k in _VIOLATION_KEYS):
        return MSG_VIOLATION
    if any(k in t for k in _SIZE_KEYS):
        return MSG_SIZE_TOO_LARGE
    return None


def _check_deps():
    if torch is None:
        raise RuntimeError("缺少必要依赖（torch）")
    if Image is None:
        raise RuntimeError("缺少必要依赖（Pillow）")


def _handle_http(e):
    raw = e.read()
    text = raw.decode("utf-8", "ignore")
    msg = _extract_api_message(raw, text)
    friendly = _categorize_text(msg) or _categorize_text(text)
    detail = (msg or text).strip()[:300]
    if friendly:
        raise RuntimeError(friendly + " | HTTP " + str(e.code) + ": " + detail)
    raise RuntimeError("生成失败 HTTP " + str(e.code) + ": " + detail)


class _ReadError(RuntimeError):
    pass


def _read_all(resp, timeout):
    buf = io.BytesIO()
    end = time.time() + timeout
    while True:
        if time.time() > end:
            raise _ReadError(MSG_TIMEOUT)
        chunk = resp.read(65536)
        if not chunk:
            return buf.getvalue()
        buf.write(chunk)


def _request(url, data, headers, timeout, method="POST", want_meta=False):
    # Preserve legacy exception classes for non-async callers.
    # Certificate failures must never retry with certificate checks disabled.
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = _read_all(resp, timeout)
            if want_meta:
                return raw, dict(resp.headers)
            return raw
    except urllib.error.HTTPError as e:
        _handle_http(e)
    except (socket.timeout, TimeoutError):
        raise RuntimeError(MSG_TIMEOUT)
    except http.client.IncompleteRead:
        raise _ReadError(MSG_TIMEOUT)
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            raise _ReadError("TLS证书验证失败，已停止请求，请检查证书或系统时间") from None
        raise RuntimeError(MSG_TIMEOUT)
    except ConnectionError:
        raise RuntimeError(MSG_GENERIC)
    except http.client.HTTPException:
        raise RuntimeError(MSG_GENERIC)


def _compress(b):
    _check_deps()
    try:
        pil = Image.open(io.BytesIO(b))
        if pil.mode in ("RGBA", "LA", "P"):
            rgba = pil.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            pil = bg
        else:
            pil = pil.convert("RGB")
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    pil.thumbnail((2048, 2048), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _bytes_to_img_tensor(data):
    _check_deps()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr)


def _img_tensor_to_bytes(img_tensor):
    _check_deps()
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        frames = [arr]
    out = []
    for fr in frames:
        a = (np.clip(fr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(a).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _extract_req_id(raw, hdrs=None):
    for key in ("x-request-id", "request-id", "x-requestid", "x-gdg-request-id", "x-goog-request-id"):
        v = (hdrs or {}).get(key)
        if v:
            return str(v).strip()[:80]
    try:
        obj = json.loads(raw.decode("utf-8"))
        rid = obj.get("responseId") or obj.get("requestId") or obj.get("id")
        if rid:
            return str(rid).strip()[:80]
    except Exception:
        pass
    return None


def _parse_result(raw):
    try:
        result = json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    b64_list = []
    for cand in result.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            d = part.get("inlineData") or part.get("inline_data") or {}
            data = d.get("data")
            if data:
                b64_list.append(data)
    if not b64_list:
        raise RuntimeError(MSG_GENERIC)
    out = []
    for b64 in b64_list:
        try:
            out.append(_bytes_to_img_tensor(base64.b64decode(b64)))
        except Exception:
            raise RuntimeError(MSG_GENERIC)
    return out


def _build_variants(seed):
    variants = []
    if seed is not None and int(seed) >= 0:
        variants.append({"quality": "high", "seed": int(seed)})
        variants.append({"seed": int(seed)})
    variants.append({"quality": "high"})
    variants.append({})
    return variants


def _generate(model, prompt, aspect, api_key, seed=-1, timeout=1200):
    url = _BASE_URL + "/v1beta/models/" + model + ":generateContent"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    req_id = None
    for extra in _build_variants(seed):
        cfg = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect}}
        cfg.update(extra)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }
        try:
            raw, hdrs = _request(url, json.dumps(payload).encode("utf-8"), headers, timeout, want_meta=True)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            if req_id:
                last = RuntimeError(str(e) + " | 请求ID: " + req_id)
            else:
                last = RuntimeError(str(e) + " | 请求ID: 无（上游未响应）")
            continue
        rid = _extract_req_id(raw, hdrs)
        if rid:
            req_id = rid
        return _parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _edits(model, prompt, aspect, ref_items, api_key, seed=-1, timeout=1200):
    url = _BASE_URL + "/v1beta/models/" + model + ":generateContent"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    req_id = None
    for extra in _build_variants(seed):
        parts = [{"text": prompt}]
        for n, b in ref_items:
            parts.append({"text": "图%d" % n})
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(b).decode("ascii"),
                }
            })
        cfg = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect}}
        cfg.update(extra)
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
        try:
            raw, hdrs = _request(url, json.dumps(payload).encode("utf-8"), headers, timeout, want_meta=True)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            if req_id:
                last = RuntimeError(str(e) + " | 请求ID: " + req_id)
            else:
                last = RuntimeError(str(e) + " | 请求ID: 无（上游未响应）")
            continue
        rid = _extract_req_id(raw, hdrs)
        if rid:
            req_id = rid
        return _parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _pack_output(tensors):
    if len(tensors) == 1:
        return tensors[0]
    h = tensors[0].shape[1]
    w = tensors[0].shape[2]
    return torch.cat(
        [t if t.shape[1] == h and t.shape[2] == w else _resize(t, h, w) for t in tensors],
        dim=0,
    )


def _resize(t, h, w):
    return torch.nn.functional.interpolate(
        t.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)


def _seed_widget():
    return ("INT", {
        "default": -1,
        "min": -1,
        "max": 0xFFFFFFFFFFFFFFFF,
        "control_after_generate": True,
        "display": "随机种子",
        "tooltip": "固定后每次生成使用相同种子；随机则每次不同。模型是否完全复现固定结果取决于接口支持。",
    })


class _Text2Image:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    SERIES = "banana_v2"
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = BANANA_ASPECT_CHOICES
    ASPECTS_EDIT = BANANA_ASPECT_CHOICES_EDIT
    ASPECT_DEFAULT = "1:1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
                "resolution": (cls.RESOLUTIONS, {"default": cls.RESOLUTIONS[0], "display": "分辨率"}),
                "aspect": (cls.ASPECTS, {"default": cls.ASPECT_DEFAULT, "display": "宽高比"}),
                "seed": _seed_widget(),
            },
        }

    def generate(self, api_key, prompt, resolution, aspect, seed=-1):
        model = _resolve_model(self.SERIES, resolution)
        tensors = _generate(model, prompt, aspect, api_key, seed=seed)
        return (_pack_output(tensors),)


class _Image2Image(_Text2Image):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = {k: v for k, v in base["required"].items() if k != "prompt"}
        required["aspect"] = (cls.ASPECTS_EDIT, {"default": cls.ASPECT_DEFAULT, "display": "宽高比"})
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {"required": required, "optional": optional}

    def generate(self, api_key, resolution, aspect, seed=-1, prompt="", **kwargs):
        model = _resolve_model(self.SERIES, resolution)
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if not slots:
            if aspect == "auto":
                aspect = "1:1"
            tensors = _generate(model, prompt, aspect, api_key, seed=seed)
        else:
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _edits(model, prompt, aspect, ref_items, api_key, seed=seed)
        return (_pack_output(tensors),)


class NineWanLiPlugin2(_Text2Image):
    SERIES = "banana_v2"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin2_1(_Image2Image):
    SERIES = "banana_v2"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin3(_Text2Image):
    SERIES = "banana_pro"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin3_1(_Image2Image):
    SERIES = "banana_pro"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPluginV2(_Text2Image):
    SERIES = "banana_v21"
    RESOLUTIONS = ["4K"]
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPluginV2_1(_Image2Image):
    SERIES = "banana_v21"
    RESOLUTIONS = ["4K"]
    DESCRIPTION = "测试节点 请勿使用"


class _SMStreamText2Image(_Text2Image):
    def generate(self, api_key, prompt, resolution, aspect, seed=-1):
        model = _resolve_model(self.SERIES, resolution)
        tensors = _sm_stream_generate(model, prompt, aspect, api_key, seed=seed)
        return (_pack_output(tensors),)


class _SMStreamImage2Image(_Image2Image):
    def generate(self, api_key, resolution, aspect, seed=-1, prompt="", **kwargs):
        model = _resolve_model(self.SERIES, resolution)
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if not slots:
            if aspect == "auto":
                aspect = "1:1"
            tensors = _sm_stream_generate(model, prompt, aspect, api_key, seed=seed)
        else:
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _sm_stream_edits(model, prompt, aspect, ref_items, api_key, seed=seed)
        return (_pack_output(tensors),)


class NineWanLiPluginStream2(_SMStreamText2Image):
    SERIES = "banana_v2"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPluginStream2_1(_SMStreamImage2Image):
    SERIES = "banana_v2"
    DESCRIPTION = "测试节点 请勿使用"


def _br_parse_result(raw):
    try:
        result = json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    data = result.get("data") or []
    item = {}
    for d in data:
        if d.get("url") or d.get("b64_json"):
            item = d
            break
    url = item.get("url")
    if url:
        last = None
        end = time.time() + 1200
        for _ in range(3):
            remain = int(end - time.time())
            if remain <= 0:
                break
            try:
                img = _br_fetch_image(url, min(120, remain))
                return [_bytes_to_img_tensor(img)]
            except RuntimeError as e:
                last = e
        raise last if last else RuntimeError(MSG_GENERIC)
    b64 = item.get("b64_json")
    if not b64:
        raise RuntimeError(MSG_GENERIC)
    try:
        return [_bytes_to_img_tensor(base64.b64decode(b64))]
    except Exception:
        raise RuntimeError(MSG_GENERIC)


def _br_build_variants(quality, seed):
    variants = []
    if seed is not None and int(seed) >= 0:
        v = {"seed": int(seed)}
        if quality:
            v["quality"] = quality
        variants.append(v)
        if quality:
            variants.append({"quality": quality})
    elif quality:
        variants.append({"quality": quality})
    variants.append({})
    return variants


def _img_generate(base_url, model, prompt, size, quality, api_key, seed=-1, timeout=1200, response_format="url"):
    url = base_url + "/v1/images/generations"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for extra in _br_build_variants(quality, seed):
        payload = {"model": model, "prompt": prompt, "size": size, "moderation": "low"}
        if response_format:
            payload["response_format"] = response_format
        payload.update(extra)
        try:
            raw = _request(url, json.dumps(payload).encode("utf-8"), headers, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        return _br_parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _sm_generate(model, prompt, size, api_key, seed=-1, timeout=1200):
    return _img_generate(_BASE_URL, model, prompt, size, None, api_key, seed=seed, timeout=timeout, response_format=None)


def _img_edits(base_url, model, prompt, size, quality, ref_items, api_key, seed=-1, timeout=1200, response_format="url"):
    url = base_url + "/v1/images/edits"
    last = None
    for variant in _br_build_variants(quality, seed):
        fields = {"model": model, "prompt": prompt, "size": size, "moderation": "low"}
        if response_format:
            fields["response_format"] = response_format
        fields.update({k: str(v) for k, v in variant.items()})
        files = [("img%d.png" % n, b) for n, b in ref_items]
        boundary, body = _br_multipart(fields, files)
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "multipart/form-data; boundary=" + boundary}
        try:
            raw = _request(url, body, headers, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        return _br_parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _sm_edits(model, prompt, size, ref_items, api_key, seed=-1, timeout=1200):
    return _img_edits(_BASE_URL, model, prompt, size, None, ref_items, api_key, seed=seed, timeout=timeout, response_format=None)


def _stream_request(url, data, headers, idle=90, total=1200):
    # Retained legacy helper; rebuilt nodes use the audited safe runtime.
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=idle) as resp:
            buf = io.BytesIO()
            end = time.time() + total
            while time.time() < end:
                try:
                    chunk = resp.read(65536)
                except socket.timeout:
                    raise _ReadError(MSG_TIMEOUT)
                if not chunk:
                    return buf.getvalue()
                buf.write(chunk)
            raise _ReadError(MSG_TIMEOUT)
    except urllib.error.HTTPError as e:
        _handle_http(e)
    except (socket.timeout, TimeoutError):
        raise _ReadError(MSG_TIMEOUT)
    except http.client.IncompleteRead:
        raise _ReadError(MSG_TIMEOUT)
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            raise _ReadError("TLS证书验证失败，已停止请求，请检查证书或系统时间") from None
        raise RuntimeError(MSG_TIMEOUT)
    except ConnectionError:
        raise RuntimeError(MSG_GENERIC)
    except http.client.HTTPException:
        raise RuntimeError(MSG_GENERIC)


def _parse_stream_images(raw):
    images = []
    try:
        text = raw.decode("utf-8", "ignore")
    except Exception:
        return images
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        try:
            cand = obj["candidates"][0]
        except Exception:
            continue
        for part in cand.get("content", {}).get("parts", []):
            if "inlineData" in part and "data" in part["inlineData"]:
                try:
                    images.append(base64.b64decode(part["inlineData"]["data"]))
                except Exception:
                    pass
    return images


def _sm_stream_generate(model, prompt, aspect, api_key, seed=-1, timeout=1200):
    last = None
    for extra in _build_variants(seed):
        cfg = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect}}
        cfg.update(extra)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }
        url = _BASE_URL + "/v1beta/models/" + model + ":streamGenerateContent"
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        try:
            raw = _stream_request(url, json.dumps(payload).encode("utf-8"), headers, idle=90, total=timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            last = e
            continue
        imgs = _parse_stream_images(raw)
        if not imgs:
            last = RuntimeError("生成失败（流式接口未返回图片）")
            continue
        tensors = []
        for b in imgs:
            try:
                tensors.append(_bytes_to_img_tensor(b))
            except Exception:
                pass
        if tensors:
            return tensors
        last = RuntimeError("生成失败（流式返回的图片无法解析）")
    raise last if last else RuntimeError(MSG_GENERIC)


def _sm_stream_edits(model, prompt, aspect, ref_items, api_key, seed=-1, timeout=1200):
    last = None
    for extra in _build_variants(seed):
        parts = [{"text": prompt}]
        for n, b in ref_items:
            parts.append({"text": "图%d" % n})
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(b).decode("ascii"),
                }
            })
        cfg = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect}}
        cfg.update(extra)
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
        url = _BASE_URL + "/v1beta/models/" + model + ":streamGenerateContent"
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        try:
            raw = _stream_request(url, json.dumps(payload).encode("utf-8"), headers, idle=90, total=timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            last = e
            continue
        imgs = _parse_stream_images(raw)
        if not imgs:
            last = RuntimeError("生成失败（流式接口未返回图片）")
            continue
        tensors = []
        for b in imgs:
            try:
                tensors.append(_bytes_to_img_tensor(b))
            except Exception:
                pass
        if tensors:
            return tensors
        last = RuntimeError("生成失败（流式返回的图片无法解析）")
    raise last if last else RuntimeError(MSG_GENERIC)


def _ref_size(t):
    s = list(t.shape)
    if len(s) == 4:
        h, w = int(s[1]), int(s[2])
    elif len(s) == 3:
        h, w = int(s[0]), int(s[1])
    else:
        return 1024, 1024
    return max(16, w), max(16, h)


def _auto_size(w, h):
    w, h = int(w), int(h)
    ratio = w / float(h)
    if ratio >= 1.0:
        W, H = 3840, int(3840 / ratio)
    else:
        W, H = int(3840 * ratio), 3840
    if W / float(H) > 3.0:
        if W >= H:
            H = int(W / 3.0)
        else:
            W = int(H / 3.0)
    pix = W * H
    if pix > 8294400:
        s = (8294400.0 / pix) ** 0.5
        W, H = int(W * s), int(H * s)
    elif pix < 655360:
        s = (655360.0 / pix) ** 0.5
        W, H = int(W * s), int(H * s)
    W = max(16, (W // 16) * 16)
    H = max(16, (H // 16) * 16)
    pix = W * H
    if pix > 8294400:
        s = (8294400.0 / pix) ** 0.5
        W = max(16, (int(W * s) // 16) * 16)
        H = max(16, (int(H * s) // 16) * 16)
    if pix < 655360:
        s = (655360.0 / pix) ** 0.5
        W = max(16, (int(W * s) // 16) * 16 + 16)
        H = max(16, (int(H * s) // 16) * 16 + 16)
    return "%dx%d" % (W, H)


class _SMText2Image:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    MODEL = None
    RESOLUTIONS = RESOLUTION_CHOICES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
                "resolution": (cls.RESOLUTIONS, {"default": cls.RESOLUTIONS[0], "display": "分辨率"}),
                "aspect": (ASPECT_CHOICES, {"default": "1:1", "display": "宽高比"}),
                "seed": _seed_widget(),
            },
        }

    def generate(self, api_key, prompt, resolution, aspect, seed=-1):
        size = _BR_SIZES[resolution][aspect]
        tensors = _sm_generate(self.MODEL, prompt, size, api_key, seed=seed)
        return (_pack_output(tensors),)


class _SMImage2Image(_SMText2Image):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = {k: v for k, v in base["required"].items() if k != "prompt"}
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {"required": required, "optional": optional}

    def generate(self, api_key, resolution, aspect, seed=-1, prompt="", **kwargs):
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if not slots:
            if aspect == "auto":
                aspect = "1:1"
            size = _BR_SIZES[resolution][aspect]
            tensors = _sm_generate(self.MODEL, prompt, size, api_key, seed=seed)
        else:
            if aspect == "auto":
                w, h = _ref_size(next(iter(slots.values())))
                size = _auto_size(w, h)
            else:
                size = _BR_SIZES[resolution][aspect]
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _sm_edits(self.MODEL, prompt, size, ref_items, api_key, seed=seed)
        return (_pack_output(tensors),)


class NineWanLiPlugin6(_SMText2Image):
    MODEL = "gpt-image-2"
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin6_1(_SMImage2Image):
    MODEL = "gpt-image-2"
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin7(_SMText2Image):
    MODEL = "gpt-image-2.5-flare"
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin7_1(_SMImage2Image):
    MODEL = "gpt-image-2.5-flare"
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"



NODE_CLASS_MAPPINGS = {
    "MuMu2-Rebuild": NineWanLiPlugin2,
    "MuMu2.1-Rebuild": NineWanLiPlugin2_1,
    "MuMu3-Rebuild": NineWanLiPlugin3,
    "MuMu3.1-Rebuild": NineWanLiPlugin3_1,
    "MuMuV2-Rebuild": NineWanLiPluginV2,
    "MuMuV2.1-Rebuild": NineWanLiPluginV2_1,
    "MuMu2-Stream-Rebuild": NineWanLiPluginStream2,
    "MuMu2.1-Stream-Rebuild": NineWanLiPluginStream2_1,
    "MuMu4-Backup-Rebuild": NineWanLiPlugin6,
    "MuMu4.1-Backup-Rebuild": NineWanLiPlugin6_1,
    "MuMu5-Backup-Rebuild": NineWanLiPlugin7,
    "MuMu5.1-Backup-Rebuild": NineWanLiPlugin7_1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuMu2-Rebuild": "木木2重制版",
    "MuMu2.1-Rebuild": "木木2.1重制版",
    "MuMu3-Rebuild": "木木3重制版",
    "MuMu3.1-Rebuild": "木木3.1重制版",
    "MuMuV2-Rebuild": "木木V2重制版",
    "MuMuV2.1-Rebuild": "木木V2.1重制版",
    "MuMu2-Stream-Rebuild": "木木2流式测试重制版",
    "MuMu2.1-Stream-Rebuild": "木木2.1流式测试重制版",
    "MuMu4-Backup-Rebuild": "木木4-备用重制版",
    "MuMu4.1-Backup-Rebuild": "木木4.1-备用重制版",
    "MuMu5-Backup-Rebuild": "木木5-备用重制版",
    "MuMu5.1-Backup-Rebuild": "木木5.1-备用重制版",
}

for _cls in [
    NineWanLiPlugin2,
    NineWanLiPlugin2_1,
    NineWanLiPlugin3,
    NineWanLiPlugin3_1,
    NineWanLiPluginV2,
    NineWanLiPluginV2_1,
    NineWanLiPluginStream2,
    NineWanLiPluginStream2_1,
    NineWanLiPlugin6,
    NineWanLiPlugin6_1,
    NineWanLiPlugin7,
    NineWanLiPlugin7_1,
]:
    _cls.NODE_NAME = NODE_DISPLAY_NAME_MAPPINGS.get(_cls.__name__, _cls.__name__)


# The safety layer keeps all legacy node IDs, widgets, models and return types.
# Import last so every legacy class resolves the audited helpers at execution.
from .safe_runtime import install as _install_mumu_safety
_install_mumu_safety(globals())

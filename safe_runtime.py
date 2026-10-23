"""Safety overrides for legacy MuMu nodes. No generation POST is retried here.

The original node classes and their wire contracts remain in nodes.py. This
module replaces the audited entry points and reference-image checks at import. Shared
helpers used by other platforms are not overwritten.
"""
import base64
import contextlib
import http.client
import io
import json
import logging
import math
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

LOG = logging.getLogger("comfy.mumu.safe")
MAX_RESPONSE_BYTES = 256 * 1024 * 1024
MAX_IMAGE_SIDE = 6500


def _redact(value):
    text = str(value or "")
    text = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", text)
    text = re.sub(r"https?://[^\s\"'<>]+", "[URL]", text)
    return text[:300]


class MuMuRequestError(RuntimeError):
    def __init__(self, message, *, status=None, request_id=None, correlation_id=None):
        self.status = status
        self.request_id = request_id
        self.correlation_id = correlation_id
        parts = [message]
        if status is not None:
            parts.append("HTTP " + str(status))
        if request_id:
            parts.append("上游请求ID: " + _redact(request_id))
        if correlation_id:
            parts.append("本地关联ID: " + correlation_id)
        super().__init__(" | ".join(parts))


class MuMuOutcomeUnknown(MuMuRequestError):
    """The paid POST may have been accepted; never submit it again implicitly."""


def _extract_req_id(raw, hdrs=None):
    lower = {str(k).lower(): str(v) for k, v in (hdrs or {}).items()}
    for key in ("x-request-id", "request-id", "x-requestid", "x-gdg-request-id", "x-goog-request-id"):
        if lower.get(key):
            return _redact(lower[key]).strip()[:100]
    try:
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            return None
        error = obj.get("error")
        error = error if isinstance(error, dict) else {}
        rid = (obj.get("responseId") or obj.get("requestId") or obj.get("request_id")
               or error.get("request_id") or error.get("requestId"))
        if rid:
            return _redact(rid).strip()[:100]
    except (ValueError, UnicodeError, AttributeError):
        pass
    return None


def _categorize_text(raw_text):
    text = str(raw_text or "").lower()
    violation = (
        "content_policy_violation", "content policy violation", "safety violation",
        "blocked by safety", "blocked by moderation", "safety filter", "prohibited content",
        "disallowed content", "sexual content", "pornographic", "内容违规", "涉嫌色情",
        "色情内容", "违反内容政策", "审核不通过", "内容审核失败",
    )
    size = (
        "image too large", "image is too large", "image dimensions", "image dimension",
        "image resolution exceeds", "image size exceeds", "maximum image size",
        "maximum image dimension", "图片尺寸过大", "图片尺寸太大", "图像尺寸过大",
        "图片分辨率过大", "图片像素过大", "超过最大分辨率",
    )
    if any(key in text for key in violation):
        return "内容违规，请修改图片或提示词后重新生成"
    if any(key in text for key in size):
        return "图片尺寸超出上游限制，请缩小图片后重试"
    return None


class _Heartbeat:
    """Comfy UI feedback only; it cannot override RH runtime/proxy limits."""
    def __init__(self):
        self.stop = threading.Event()
        self.thread = None
        self.progress = None

    def __enter__(self):
        try:
            from comfy.utils import ProgressBar
            self.progress = ProgressBar(1000)
        except (ImportError, AttributeError):
            return self
        def beat():
            tick = 0
            while not self.stop.wait(10):
                tick = min(999, tick + 1)
                try:
                    self.progress.update_absolute(tick, 1000)
                except Exception:
                    LOG.debug("MuMu progress feedback unavailable", exc_info=False)
                    return
        self.thread = threading.Thread(target=beat, name="mumu-ui-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if exc_type is None and self.progress is not None:
            try:
                self.progress.update_absolute(1000, 1000)
            except Exception:
                pass


class _SafeRuntime:
    def __init__(self, namespace):
        self.ns = namespace

    @staticmethod
    def _set_read_timeout(resp, remaining):
        try:
            resp.fp.raw._sock.settimeout(max(0.001, remaining))
        except (AttributeError, OSError):
            pass

    def read_response(self, resp, deadline):
        body = io.BytesIO()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("response deadline reached")
            self._set_read_timeout(resp, remaining)
            reader = getattr(resp, "read1", resp.read)
            chunk = reader(65536)
            if not chunk:
                return body.getvalue()
            body.write(chunk)
            if body.tell() > MAX_RESPONSE_BYTES:
                raise ValueError("上游响应超过256MiB，已停止读取")

    def request(self, url, data, headers, timeout, method="POST", want_meta=False):
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout必须是正数")
        method = method.upper()
        paid = method == "POST"
        correlation = uuid.uuid4().hex
        started = time.monotonic()
        request_headers = dict(headers)
        request_headers.setdefault("X-Client-Request-Id", correlation)
        req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        # The default TLS context validates both the certificate chain and host.
        rid = None
        endpoint = urllib.parse.urlsplit(url).path
        LOG.info("MuMu stage=request_start method=%s endpoint=%s correlation=%s", method, endpoint, correlation)
        try:
            heartbeat = _Heartbeat() if paid else contextlib.nullcontext()
            with heartbeat:
                with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
                    hdrs = dict(resp.headers)
                    rid = _extract_req_id(b"", hdrs)
                    LOG.info("MuMu stage=response_headers method=%s elapsed=%.3f status=%s request_id=%s correlation=%s",
                             method, time.monotonic() - started, getattr(resp, "status", None), rid, correlation)
                    raw = self.read_response(resp, started + timeout)
                    rid = rid or _extract_req_id(raw, hdrs)
                    LOG.info("MuMu stage=response_complete method=%s elapsed=%.3f bytes=%d request_id=%s correlation=%s",
                             method, time.monotonic() - started, len(raw), rid, correlation)
                    return (raw, hdrs) if want_meta else raw
        except urllib.error.HTTPError as exc:
            rid = _extract_req_id(b"", dict(exc.headers or {}))
            try:
                try:
                    raw = exc.read(16384)
                except (OSError, http.client.HTTPException):
                    raw = b""
            finally:
                exc.close()
            rid = rid or _extract_req_id(raw)
            try:
                obj = json.loads(raw.decode("utf-8"))
                err = obj.get("error", obj)
                detail = err.get("message", "上游HTTP请求失败") if isinstance(err, dict) else str(err)
            except (ValueError, UnicodeError, AttributeError):
                detail = "上游HTTP请求失败"
            friendly = _categorize_text(detail)
            message = friendly or _redact(detail)
            LOG.warning("MuMu stage=http_error method=%s elapsed=%.3f status=%s request_id=%s correlation=%s",
                        method, time.monotonic() - started, exc.code, rid, correlation)
            if paid and exc.code >= 500:
                message += "；任务结果未知，上游可能仍在生成或计费，未自动重提，请先核对上游记录"
            raise MuMuRequestError(message, status=exc.code, request_id=rid, correlation_id=correlation) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                http.client.HTTPException, OSError, ValueError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if isinstance(reason, ssl.SSLCertVerificationError):
                message = "TLS证书验证失败，已停止请求，请检查证书或系统时间"
                error_class = MuMuRequestError
            elif paid:
                message = "连接中断或任务超时，结果未知；上游可能仍在生成或计费，未自动重提，请先核对上游记录"
                error_class = MuMuOutcomeUnknown
            else:
                message = "图片/状态GET请求失败（" + type(reason).__name__ + "）"
                error_class = MuMuRequestError
            LOG.warning("MuMu stage=transport_error method=%s elapsed=%.3f type=%s request_id=%s correlation=%s",
                        method, time.monotonic() - started, type(reason).__name__, rid, correlation)
            raise error_class(message, request_id=rid, correlation_id=correlation) from None

    def bytes_to_tensor(self, data):
        ns = self.ns
        ns["_check_deps"]()
        if not isinstance(data, bytes) or not data:
            raise RuntimeError("图片数据为空或类型错误，未获得可用图片")
        try:
            with ns["Image"].open(io.BytesIO(data)) as original:
                if original.width <= 0 or original.height <= 0:
                    raise ValueError("empty dimensions")
                original.load()
                image = original.convert("RGB")
            arr = ns["np"].array(image, dtype=ns["np"].float32) / 255.0
            return ns["torch"].from_numpy(arr[None, ...])
        except Exception as exc:
            raise RuntimeError("返回内容不是完整可解码的图片（" + type(exc).__name__ + "）") from None

    def tensor_to_bytes(self, tensor):
        ns = self.ns
        ns["_check_deps"]()
        if tensor is None or not hasattr(tensor, "detach"):
            raise RuntimeError("参考图为空或不是ComfyUI IMAGE，请检查上传及上游图片节点")
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0 or arr.shape[3] not in (1, 3, 4):
            raise RuntimeError("参考图tensor为空或形状错误，要求[B,H,W,C]且C=1/3/4，未提交生成")
        if arr.shape[0] != 1:
            raise RuntimeError("一个参考图插口收到多帧batch；请先拆分为单帧，避免静默丢失参考图，未提交生成")
        if max(arr.shape[1:3]) > MAX_IMAGE_SIDE:
            raise RuntimeError("图片尺寸太大，请将最长边调整到6500像素以下，未提交生成")
        if not ns["np"].isfinite(arr).all():
            raise RuntimeError("参考图包含NaN或无穷值，未提交生成")
        outputs = []
        for frame in arr:
            pixels = (ns["np"].clip(frame, 0, 1) * 255).round().astype(ns["np"].uint8)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            image = ns["Image"].fromarray(pixels)
            if image.mode == "RGBA":
                bg = ns["Image"].new("RGB", image.size, "white")
                bg.paste(image, mask=image.getchannel("A"))
                image = bg
            else:
                image = image.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            outputs.append(buf.getvalue())
        return outputs

    def validate_reference_bytes(self, ref_items):
        if not ref_items:
            raise RuntimeError("参考图为空，未提交图生图；请检查网站到RH的图片参数绑定")
        for number, data in ref_items:
            if not isinstance(data, bytes) or not data:
                raise RuntimeError("图%d数据为空，未提交生成" % number)
            try:
                with self.ns["Image"].open(io.BytesIO(data)) as image:
                    if image.format != "PNG":
                        raise ValueError("参考图必须是PNG，以匹配声明的MIME")
                    if max(image.size) > MAX_IMAGE_SIDE:
                        raise ValueError("最长边超过6500像素")
                    image.verify()
            except Exception as exc:
                raise RuntimeError("图%d无法作为有效PNG上传：%s，未提交生成" % (number, _redact(exc))) from None

    @staticmethod
    def validate_request(api_key, prompt):
        if not isinstance(api_key, str) or not api_key.strip():
            raise RuntimeError("缺少API密钥，未提交生成")
        if not isinstance(prompt, str):
            raise RuntimeError("提示词类型错误，未提交生成")

    def parse_gemini(self, raw):
        try:
            result = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise RuntimeError("上游已响应，但返回格式不是JSON；未自动重提，请核对上游记录") from None
        if not isinstance(result, dict):
            raise RuntimeError("上游响应结构错误；未自动重提，请核对上游记录")
        self._raise_api_error(result)
        outputs = []
        for candidate in result.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            for part in (candidate.get("content") or {}).get("parts") or []:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data") or {}
                encoded = inline.get("data")
                if encoded:
                    try:
                        data = base64.b64decode(encoded, validate=True)
                    except (ValueError, TypeError):
                        raise RuntimeError("上游图片base64损坏；未自动重提，请核对上游记录") from None
                    outputs.append(self.bytes_to_tensor(data))
        if not outputs:
            feedback = result.get("promptFeedback") or result.get("prompt_feedback") or {}
            reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
            finishes = [c.get("finishReason") for c in result.get("candidates", []) if isinstance(c, dict) and c.get("finishReason")]
            detail = _redact(reason or ",".join(finishes) or "未提供原因")
            raise RuntimeError("上游未返回图片（" + detail + "）；未自动重提，请核对上游记录")
        return outputs

    @staticmethod
    def _raise_api_error(result):
        error = result.get("error")
        if error:
            detail = error.get("message", error.get("code", "上游返回错误")) if isinstance(error, dict) else str(error)
            raise MuMuRequestError(_categorize_text(detail) or _redact(detail), request_id=_extract_req_id(json.dumps(result).encode()))

    def download_image(self, url, timeout):
        parsed = urllib.parse.urlsplit(str(url))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise RuntimeError("上游返回的图片URL无效")
        deadline = time.monotonic() + float(timeout)
        last = None
        for attempt in range(3):
            remain = deadline - time.monotonic()
            if remain <= 0:
                break
            try:
                raw = self.request(url, None, {"User-Agent": "ComfyUI-Node/1.0", "Accept": "image/*"}, min(120, remain), method="GET")
                # Verify before retry decisions. No new image generation is submitted.
                self.bytes_to_tensor(raw)
                return raw
            except RuntimeError as exc:
                last = exc
                if isinstance(exc, MuMuRequestError) and exc.status is not None and 400 <= exc.status < 500 and exc.status not in (408, 429):
                    break
                if attempt < 2:
                    time.sleep(min(0.5 * (2 ** attempt), max(0, deadline - time.monotonic())))
        raise RuntimeError("生成接口已返回，但结果图片下载失败；仅重试了GET，未重新生成。" + _redact(last)) from None

    def parse_openai(self, raw):
        try:
            result = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise RuntimeError("上游响应不是JSON；未自动重提，请核对上游记录") from None
        if not isinstance(result, dict):
            raise RuntimeError("上游响应结构错误；未自动重提")
        self._raise_api_error(result)
        outputs = []
        for item in result.get("data") or []:
            if not isinstance(item, dict):
                continue
            if item.get("b64_json"):
                try:
                    data = base64.b64decode(item["b64_json"], validate=True)
                except (ValueError, TypeError):
                    raise RuntimeError("上游图片base64损坏；未自动重提") from None
            elif item.get("url"):
                data = self.download_image(item["url"], 360)
            else:
                continue
            outputs.append(self.bytes_to_tensor(data))
        if not outputs:
            raise RuntimeError("上游未返回可用图片；未自动重提，请核对上游记录")
        return outputs

    def gemini(self, model, prompt, aspect, api_key, seed=-1, timeout=1200, ref_items=None, stream=False):
        self.validate_request(api_key, prompt)
        parts = [{"text": prompt}]
        if ref_items is not None:
            self.validate_reference_bytes(ref_items)
            for number, data in ref_items:
                parts += [{"text": "图%d" % number}, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(data).decode("ascii")}}]
        elif not prompt.strip():
            raise RuntimeError("提示词为空，未提交文生图")
        config = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect}, "quality": "high"}
        if seed is not None and int(seed) >= 0:
            config["seed"] = int(seed)
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        endpoint = ":streamGenerateContent" if stream else ":generateContent"
        url = self.ns["_BASE_URL"] + "/v1beta/models/" + model + endpoint
        headers = {"Authorization": "Bearer " + api_key.strip(), "Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"
        raw, meta = self.request(url, json.dumps(payload).encode("utf-8"), headers, timeout, want_meta=True)
        try:
            return self.parse_stream(raw) if stream else self.parse_gemini(raw)
        except RuntimeError as exc:
            rid = _extract_req_id(raw, meta)
            if rid:
                raise RuntimeError(str(exc) + " | 上游请求ID: " + rid) from None
            raise

    def parse_stream(self, raw):
        text = raw.decode("utf-8", "replace")
        if text.lstrip().startswith("{"):
            # Some gateways respond with a regular synchronous JSON object.
            return self.parse_gemini(raw)
        objects = []
        event_lines = []
        for line in text.replace("\r\n", "\n").split("\n") + [""]:
            if line.startswith("data:"):
                event_lines.append(line[5:].lstrip())
            elif not line and event_lines:
                data = "\n".join(event_lines)
                event_lines = []
                if data.strip() == "[DONE]":
                    continue
                try:
                    objects.append(json.loads(data))
                except ValueError:
                    raise RuntimeError("流式事件不是有效JSON，未自动重提") from None
        if not objects:
            raise RuntimeError("流式接口没有返回可用事件，未自动重提；请使用已确认支持的同步节点")
        merged = {"candidates": []}
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            self._raise_api_error(obj)
            merged["candidates"].extend(obj.get("candidates") or [])
            if obj.get("promptFeedback"):
                merged["promptFeedback"] = obj["promptFeedback"]
        return self.parse_gemini(json.dumps(merged).encode("utf-8"))

    def openai_generate(self, base_url, model, prompt, size, quality, api_key, seed=-1, timeout=1200, response_format="url"):
        self.validate_request(api_key, prompt)
        if not prompt.strip():
            raise RuntimeError("提示词为空，未提交文生图")
        payload = {"model": model, "prompt": prompt, "size": size, "moderation": "low"}
        if quality:
            payload["quality"] = quality
        if seed is not None and int(seed) >= 0:
            payload["seed"] = int(seed)
        if response_format:
            payload["response_format"] = response_format
        raw, meta = self.request(base_url + "/v1/images/generations", json.dumps(payload).encode(),
                                 {"Authorization": "Bearer " + api_key.strip(), "Content-Type": "application/json"}, timeout, want_meta=True)
        return self._parse_openai_with_id(raw, meta)

    def _parse_openai_with_id(self, raw, meta):
        try:
            return self.parse_openai(raw)
        except RuntimeError as exc:
            rid = _extract_req_id(raw, meta)
            if rid:
                raise RuntimeError(str(exc) + " | 上游请求ID: " + rid) from None
            raise

    def openai_edits(self, base_url, model, prompt, size, quality, ref_items, api_key, seed=-1, timeout=1200, response_format="url"):
        self.validate_request(api_key, prompt)
        self.validate_reference_bytes(ref_items)
        fields = {"model": model, "prompt": prompt, "size": size, "moderation": "low"}
        if quality:
            fields["quality"] = quality
        if seed is not None and int(seed) >= 0:
            fields["seed"] = int(seed)
        if response_format:
            fields["response_format"] = response_format
        boundary, body = self.ns["_br_multipart"](fields, [("img%d.png" % n, data) for n, data in ref_items])
        raw, meta = self.request(base_url + "/v1/images/edits", body,
                                 {"Authorization": "Bearer " + api_key.strip(), "Content-Type": "multipart/form-data; boundary=" + boundary}, timeout, want_meta=True)
        return self._parse_openai_with_id(raw, meta)


def install(namespace):
    """Preserve registration and widget signatures; change the called helpers."""
    runtime = _SafeRuntime(namespace)
    namespace["uuid"] = uuid  # Legacy multipart builder omitted this import.
    # Do not replace the legacy shared transport or tensor helpers: other
    # callers use their exception types to control asynchronous resubmission.

    def generate(model, prompt, aspect, api_key, seed=-1, timeout=1200):
        return runtime.gemini(model, prompt, aspect, api_key, seed, timeout)
    def edits(model, prompt, aspect, ref_items, api_key, seed=-1, timeout=1200):
        return runtime.gemini(model, prompt, aspect, api_key, seed, timeout, ref_items)
    def stream_generate(model, prompt, aspect, api_key, seed=-1, timeout=1200):
        return runtime.gemini(model, prompt, aspect, api_key, seed, timeout, stream=True)
    def stream_edits(model, prompt, aspect, ref_items, api_key, seed=-1, timeout=1200):
        return runtime.gemini(model, prompt, aspect, api_key, seed, timeout, ref_items, True)
    namespace["_generate"] = generate
    namespace["_edits"] = edits
    namespace["_sm_stream_generate"] = stream_generate
    namespace["_sm_stream_edits"] = stream_edits
    def sm_generate(model, prompt, size, api_key, seed=-1, timeout=1200):
        return runtime.openai_generate(namespace["_BASE_URL"], model, prompt, size, None, api_key,
                                       seed=seed, timeout=timeout, response_format=None)
    def sm_edits(model, prompt, size, ref_items, api_key, seed=-1, timeout=1200):
        return runtime.openai_edits(namespace["_BASE_URL"], model, prompt, size, None, ref_items, api_key,
                                    seed=seed, timeout=timeout, response_format=None)
    namespace["_sm_generate"] = sm_generate
    namespace["_sm_edits"] = sm_edits

    # Before any paid image-to-image request, require at least one real IMAGE.
    def image_method(*, stream=False, openai=False):
        def checked_generate(self, api_key, resolution, aspect, seed=-1, prompt="", **kwargs):
            slots = [(i + 1, kwargs.get(key)) for i, key in enumerate(namespace["_IMAGE_KEYS"]) if kwargs.get(key) is not None]
            if not slots:
                raise RuntimeError("图生图参考图为空，未提交生成；请检查网站上传、RH参数绑定及上游图片节点")
            refs = [(number, runtime.tensor_to_bytes(tensor)[0]) for number, tensor in slots]
            if openai:
                if aspect == "auto":
                    width, height = namespace["_ref_size"](slots[0][1])
                    size = namespace["_auto_size"](width, height)
                else:
                    size = namespace["_BR_SIZES"][resolution][aspect]
                tensors = sm_edits(self.MODEL, prompt, size, refs, api_key, seed=seed)
            else:
                model = namespace["_resolve_model"](self.SERIES, resolution)
                tensors = runtime.gemini(model, prompt, aspect, api_key, seed=seed, ref_items=refs, stream=stream)
            return (namespace["_pack_output"](tensors),)
        return checked_generate
    namespace["_Image2Image"].generate = image_method()
    namespace["_SMStreamImage2Image"].generate = image_method(stream=True)
    namespace["_SMImage2Image"].generate = image_method(openai=True)
    namespace["_mumu_safety_runtime"] = runtime
    namespace["MUMU_SAFETY_VERSION"] = "2026.10.23.2"

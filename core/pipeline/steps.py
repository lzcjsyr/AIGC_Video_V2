"""流水线分步执行实现模块。

本文件承载步骤 1~6 的核心实现与跨步骤辅助函数，主要包括：
1. 项目初始化与目录/文件落盘（raw/script/keywords/音视频资源）。
2. 文本处理链路（摘要、分段脚本、关键词与描述摘要生成）。
3. 多媒体生成链路（开场图、分段配图、语音合成、最终视频合成、封面图生成）。
4. 运行时工具函数（路径解析、开场旁白生成、BGM 定位、失败兜底处理）。

设计上保持“单步可独立调用”，便于 CLI 分步重跑、API 精细化控制与测试隔离。
"""

import datetime
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.config import Config, config
from core.domain.composer import VideoComposer
from core.domain.docx_transform import export_raw_to_docx
from core.domain.reader import DocumentReader
from core.infra.ai.image_client import (
    generate_cover_images,
    generate_images_for_segments,
    generate_opening_image,
    synthesize_voice_for_segments,
)
from core.domain.summarizer import (
    export_plain_text_segments,
    extract_keywords,
    generate_description_summary,
    intelligent_summarize,
    parse_json_robust,
    process_raw_to_script,
)
from core.infra.ai import text_to_audio_bytedance
from core.infra.media.youtube_client import (
    download_youtube_video,
    is_valid_youtube_url,
    probe_video_duration,
    search_youtube_candidates,
    trim_video_clip,
)
from core.infra.project_paths import ProjectPaths
from core.llm_gateway import text_to_text
from core.prompts import (
    YOUTUBE_MEDIA_OPENING_PLAN_PROMPT_TEMPLATE,
    YOUTUBE_MEDIA_SEGMENT_PLAN_PROMPT_TEMPLATE,
    YOUTUBE_MEDIA_SEGMENT_PLAN_SYSTEM_PROMPT,
)
from core.shared import load_json_file, logger

SEGMENT_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v")


def _get_project_root() -> str:
    """Return project root path regardless of package nesting depth."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _initialize_project(raw_data: Dict[str, Any], output_dir: str) -> tuple:
    """Create project folder structure and persist raw outputs."""
    current_time = datetime.datetime.now()
    time_suffix = current_time.strftime("%m%d_%H%M")
    raw_title = raw_data.get("title", "untitled") or "untitled"
    project_folder = f"{raw_title}_{time_suffix}"
    project_output_dir = os.path.join(output_dir, project_folder)

    paths = ProjectPaths(project_output_dir)
    paths.ensure_dirs_exist()

    with open(paths.raw_json(), "w", encoding="utf-8") as handle:
        json.dump(raw_data, handle, ensure_ascii=False, indent=2)

    raw_docx_path = None
    try:
        export_raw_to_docx(raw_data, paths.raw_docx())
        raw_docx_path = paths.raw_docx()
    except Exception:
        pass

    return project_output_dir, paths.raw_json(), raw_docx_path


def _resolve_bgm_audio_path(bgm_filename: Optional[str], project_root: str) -> Optional[str]:
    """Locate BGM asset either via absolute path or music directory."""
    if not bgm_filename:
        return None
    if os.path.isabs(bgm_filename) and os.path.exists(bgm_filename):
        return bgm_filename
    candidate = os.path.join(project_root, "music", bgm_filename)
    if os.path.exists(candidate):
        return candidate
    return None


def _ensure_opening_narration(
    script_data: Optional[Dict[str, Any]],
    voice_dir: str,
    voice: str,
    opening_quote: bool,
    announce: bool = False,
    force_regenerate: bool = False,
    speech_rate: int = 0,
    loudness_rate: int = 0,
    emotion: str = "neutral",
    emotion_scale: int = 4,
    mute_cut_threshold: int = 400,
    mute_cut_min_silence_ms: int = 200,
    mute_cut_remain_ms: int = 100,
) -> Optional[str]:
    """Generate or reuse opening narration audio when required."""
    opening_golden_quote = (script_data or {}).get("golden_quote", "")
    if not (opening_quote and isinstance(opening_golden_quote, str) and opening_golden_quote.strip()):
        return None

    try:
        os.makedirs(voice_dir, exist_ok=True)
        opening_path = os.path.join(voice_dir, "opening.mp3")
        if force_regenerate and os.path.exists(opening_path):
            try:
                os.remove(opening_path)
            except Exception:
                if announce:
                    print("⚠️ 开场音频删除失败，尝试直接覆盖")

        if os.path.exists(opening_path):
            if announce:
                print(f"✅ 开场音频已存在: {opening_path}")
            return opening_path

        ok = text_to_audio_bytedance(
            opening_golden_quote,
            opening_path,
            voice=voice,
            encoding="mp3",
            speech_rate=speech_rate,
            loudness_rate=loudness_rate,
            emotion=emotion,
            emotion_scale=emotion_scale,
            mute_cut_threshold=mute_cut_threshold,
            mute_cut_min_silence_ms=mute_cut_min_silence_ms,
            mute_cut_remain_ms=mute_cut_remain_ms,
        )
        if ok:
            if announce:
                print(f"✅ 开场音频已生成: {opening_path}")
            return opening_path
        if announce:
            print("❌ 开场音频生成失败")
    except Exception:
        if announce:
            print("❌ 开场音频生成失败")
    return None


def _invoke_opening_narration(
    script_data: Optional[Dict[str, Any]],
    voice_dir: str,
    voice: str,
    opening_quote: bool,
    *,
    announce: bool = False,
    force_regenerate: bool = False,
    speech_rate: int = 0,
    loudness_rate: int = 0,
    emotion: str = "neutral",
    emotion_scale: int = 4,
    mute_cut_threshold: int = 400,
    mute_cut_min_silence_ms: int = 200,
    mute_cut_remain_ms: int = 100,
) -> Optional[str]:
    """Call _ensure_opening_narration with graceful fallback for legacy mocks."""
    func = _ensure_opening_narration
    try:
        return func(
            script_data,
            voice_dir,
            voice,
            opening_quote,
            announce=announce,
            force_regenerate=force_regenerate,
            speech_rate=speech_rate,
            loudness_rate=loudness_rate,
            emotion=emotion,
            emotion_scale=emotion_scale,
            mute_cut_threshold=mute_cut_threshold,
            mute_cut_min_silence_ms=mute_cut_min_silence_ms,
            mute_cut_remain_ms=mute_cut_remain_ms,
        )
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" in message and (
            "speech_rate" in message or "loudness_rate" in message
        ):
            return func(
                script_data,
                voice_dir,
                voice,
                opening_quote,
                announce=announce,
                force_regenerate=force_regenerate,
            )
        raise


def _resolve_description_source_text(
    project_output_dir: str,
    raw_data: Optional[Dict[str, Any]] = None,
    script_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Prefer raw.docx edits when building description-mode summary input."""
    docx_path = os.path.join(project_output_dir, "text", "raw.docx")
    if os.path.exists(docx_path):
        try:
            from core.domain.docx_transform import parse_raw_from_docx

            parsed = parse_raw_from_docx(docx_path)
            content = (parsed.get("content") or "").strip()
            if content:
                return content
        except Exception as exc:
            logger.warning(f"解析raw.docx失败，改用备用内容: {exc}")

    if raw_data:
        content = (raw_data.get("content") or "").strip()
        if content:
            return content

    if script_data:
        segments = script_data.get("segments") or []
        merged = "\n".join(seg.get("content", "") for seg in segments).strip()
        if merged:
            return merged

    return ""


def _resolve_segment_media_path(paths: ProjectPaths, index: int) -> Optional[str]:
    """解析段落素材路径：优先图片，其次视频（兼容扩展名大小写）。"""
    png_path = paths.segment_image(index)
    if os.path.exists(png_path):
        return png_path

    base_name = f"segment_{index}"

    # 优先常见小写命名
    for ext in SEGMENT_VIDEO_EXTENSIONS:
        candidate = os.path.join(paths.media, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate

    # 回退：扫描目录，兼容 .MP4/.MOV 等大小写扩展名
    try:
        for filename in os.listdir(paths.media):
            file_path = os.path.join(paths.media, filename)
            if not os.path.isfile(file_path):
                continue
            stem, ext = os.path.splitext(filename)
            if stem.lower() == base_name and ext.lower() in SEGMENT_VIDEO_EXTENSIONS:
                return file_path
    except OSError:
        pass

    return None


def _write_json_file(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_interval(
    start_seconds: Any,
    end_seconds: Any,
    *,
    candidate_duration: float,
    target_duration: float,
) -> Tuple[float, float]:
    start = max(0.0, _safe_float(start_seconds, 0.0))
    end = _safe_float(end_seconds, 0.0)
    duration = max(0.1, target_duration)

    if end <= start:
        end = start + duration

    if candidate_duration > 0:
        end = min(end, candidate_duration)
        if end <= start:
            start = max(0.0, candidate_duration - duration)
            end = min(candidate_duration, start + duration)

    if end <= start:
        end = start + duration

    return round(start, 3), round(end, 3)


def _build_segment_query(segment: Dict[str, Any], keyword_item: Optional[Dict[str, Any]] = None) -> str:
    keywords = (keyword_item or {}).get("keywords") or []
    atmos = (keyword_item or {}).get("atmosphere") or []
    tokens = [str(token).strip() for token in [*keywords, *atmos] if str(token).strip()]
    if tokens:
        return " ".join(tokens[:6])

    content = str(segment.get("content") or "").strip()
    return content[:60]


def _build_opening_query(script_data: Dict[str, Any]) -> str:
    title = str(script_data.get("title") or "").strip()
    content_title = str(script_data.get("content_title") or "").strip()
    golden_quote = str(script_data.get("golden_quote") or "").strip()
    values = [value for value in [golden_quote, content_title, title] if value]
    return " ".join(values)[:120]


def _validate_media_plan_choice(
    plan_payload: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    target_duration: float,
) -> Dict[str, Any]:
    fallback_candidate = candidates[0] if candidates else {}
    allowed_urls = {item.get("youtube_url") for item in candidates if item.get("youtube_url")}
    picked_url = str(plan_payload.get("youtube_url") or "").strip()
    if picked_url not in allowed_urls:
        picked_url = str(fallback_candidate.get("youtube_url") or "").strip()

    selected = next((item for item in candidates if item.get("youtube_url") == picked_url), fallback_candidate)
    candidate_duration = _safe_float(selected.get("duration"), 0.0)
    default_duration = max(0.1, target_duration)
    if candidate_duration > 0:
        default_duration = min(default_duration, candidate_duration)

    start, end = _clip_interval(
        plan_payload.get("start_seconds", 0.0),
        plan_payload.get("end_seconds", default_duration),
        candidate_duration=candidate_duration,
        target_duration=default_duration,
    )

    reason = str(plan_payload.get("reason") or "").strip()
    return {
        "youtube_url": picked_url,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": round(max(0.0, end - start), 3),
        "reason": reason,
    }


def _plan_with_llm(
    *,
    llm_server: str,
    llm_model: str,
    prompt: str,
) -> Dict[str, Any]:
    try:
        response = text_to_text(
            server=llm_server,
            model=llm_model,
            prompt=prompt,
            system_message=YOUTUBE_MEDIA_SEGMENT_PLAN_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.2,
        )
        if not response:
            return {}
        parsed = parse_json_robust(response)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        logger.warning(f"YouTube规划解析失败，使用回退策略: {exc}")
        return {}


def _build_media_plan(
    *,
    llm_server: str,
    llm_model: str,
    script_data: Dict[str, Any],
    visual_mode: str,
    keywords_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    search_top_k = int(getattr(config, "YOUTUBE_SEARCH_TOP_K", 8))
    min_duration = int(getattr(config, "YOUTUBE_MIN_DURATION_SECONDS", 6))
    max_duration = int(getattr(config, "YOUTUBE_MAX_DURATION_SECONDS", 180))
    default_clip_duration = float(getattr(config, "YOUTUBE_DEFAULT_CLIP_DURATION_SECONDS", 6.5))
    opening_default_duration = float(getattr(config, "YOUTUBE_OPENING_DEFAULT_DURATION_SECONDS", 6.5))

    keyword_segments = (keywords_data or {}).get("segments") or []
    segments = script_data.get("segments") or []
    planned_segments: List[Dict[str, Any]] = []

    for index, segment in enumerate(segments, 1):
        keyword_item = keyword_segments[index - 1] if index - 1 < len(keyword_segments) else {}
        query = _build_segment_query(segment, keyword_item)
        try:
            candidates = search_youtube_candidates(
                query,
                limit=search_top_k,
                min_duration_seconds=min_duration,
                max_duration_seconds=max_duration,
            )
        except Exception as exc:
            logger.warning(f"第{index}段YouTube候选检索失败: {exc}")
            candidates = []
        estimated_duration = max(0.1, _safe_float(segment.get("estimated_duration"), default_clip_duration))

        llm_choice: Dict[str, Any] = {}
        if candidates:
            prompt = YOUTUBE_MEDIA_SEGMENT_PLAN_PROMPT_TEMPLATE.format(
                segment_index=index,
                segment_content=str(segment.get("content") or ""),
                target_duration=round(estimated_duration, 3),
                candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
            )
            llm_choice = _plan_with_llm(llm_server=llm_server, llm_model=llm_model, prompt=prompt)

        normalized = _validate_media_plan_choice(
            llm_choice,
            candidates,
            target_duration=estimated_duration,
        )

        planned_segments.append(
            {
                "index": index,
                "source": "youtube" if visual_mode == "youtube" else "image",
                "estimated_duration": round(estimated_duration, 3),
                "query": query,
                "youtube_url": normalized["youtube_url"],
                "start_seconds": normalized["start_seconds"],
                "end_seconds": normalized["end_seconds"],
                "duration_seconds": normalized["duration_seconds"],
                "reason": normalized["reason"],
                "candidates": candidates,
            }
        )

    opening_query = _build_opening_query(script_data)
    try:
        opening_candidates = search_youtube_candidates(
            opening_query,
            limit=search_top_k,
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
        )
    except Exception as exc:
        logger.warning(f"开场YouTube候选检索失败: {exc}")
        opening_candidates = []
    opening_choice: Dict[str, Any] = {}
    if opening_candidates:
        opening_text = str(script_data.get("golden_quote") or script_data.get("title") or "")
        opening_prompt = YOUTUBE_MEDIA_OPENING_PLAN_PROMPT_TEMPLATE.format(
            opening_text=opening_text,
            target_duration=round(opening_default_duration, 3),
            candidates_json=json.dumps(opening_candidates, ensure_ascii=False, indent=2),
        )
        opening_choice = _plan_with_llm(llm_server=llm_server, llm_model=llm_model, prompt=opening_prompt)

    opening_normalized = _validate_media_plan_choice(
        opening_choice,
        opening_candidates,
        target_duration=opening_default_duration,
    )

    return {
        "version": "1.0",
        "visual_mode": visual_mode,
        "created_time": datetime.datetime.now().isoformat(),
        "opening": {
            "source": "youtube" if visual_mode in {"youtube", "hybrid"} else "image",
            "query": opening_query,
            "youtube_url": opening_normalized["youtube_url"],
            "start_seconds": opening_normalized["start_seconds"],
            "end_seconds": opening_normalized["end_seconds"],
            "duration_seconds": opening_normalized["duration_seconds"],
            "reason": opening_normalized["reason"],
            "candidates": opening_candidates,
        },
        "segments": planned_segments,
    }


def run_step_1(
    input_file: str,
    output_dir: str,
    llm_server: str,
    llm_model: str,
    target_length: int,
    num_segments: int,
) -> Dict[str, Any]:
    reader = DocumentReader()
    document_content, _ = reader.read(input_file)
    raw_data = intelligent_summarize(llm_server, llm_model, document_content, target_length, num_segments)

    project_output_dir, raw_json_path, raw_docx_path = _initialize_project(raw_data, output_dir)

    return {
        "success": True,
        "project_output_dir": project_output_dir,
        "raw": {
            "raw_json_path": raw_json_path,
            "raw_docx_path": raw_docx_path,
            "total_length": raw_data.get("total_length", 0),
        },
    }


def run_step_1_5(
    project_output_dir: str,
    num_segments: int,
    is_new_project: bool = False,
    raw_data: Optional[Dict[str, Any]] = None,
    auto_mode: bool = False,
    split_mode: str = "auto",
) -> Dict[str, Any]:
    _ = auto_mode

    try:
        print("正在处理原始内容为脚本...")

        paths = ProjectPaths(project_output_dir)
        raw_json_path = paths.raw_json()
        raw_docx_path = paths.raw_docx()
        script_path = paths.script_json()
        script_docx_path = paths.script_docx()

        if is_new_project and raw_data is not None:
            logger.info("新建项目：使用提供的raw数据")
            current_raw_data = raw_data
        else:
            if not os.path.exists(raw_json_path):
                current_raw_data = {"title": "手动创建项目", "golden_quote": "", "content": "", "target_segments": num_segments}
            else:
                print(f"加载raw数据: {raw_json_path}")
                current_raw_data = load_json_file(raw_json_path)
                if current_raw_data is None:
                    return {"success": False, "message": f"无法加载 raw.json 文件: {raw_json_path}"}
                old_segments = current_raw_data.get("target_segments")
                if old_segments and old_segments != num_segments:
                    print(f"检测到分段数变更: {old_segments} → {num_segments}")
                print(f"当前分段数: {num_segments}")

        updated_raw_data = current_raw_data
        if os.path.exists(raw_docx_path):
            try:
                from core.domain.docx_transform import parse_raw_from_docx, export_script_to_docx

                parsed_data = parse_raw_from_docx(raw_docx_path)
                if parsed_data is not None:
                    print("已从编辑后的DOCX文件解析内容")
                    updated_raw_data = parsed_data
                    updated_raw_data.update(
                        {
                            "target_segments": num_segments,
                            "created_time": current_raw_data.get("created_time"),
                            "model_info": current_raw_data.get("model_info", {}),
                            "total_length": len(updated_raw_data.get("content", "")),
                        }
                    )
                    with open(raw_json_path, "w", encoding="utf-8") as handle:
                        json.dump(updated_raw_data, handle, ensure_ascii=False, indent=2)
                    print(f"已更新原始JSON: {raw_json_path}")
                else:
                    print("⚠️  DOCX解析返回None，使用原始数据")
            except Exception as exc:
                print(f"⚠️  解析DOCX失败，使用原始数据: {exc}")
                from core.domain.docx_transform import export_script_to_docx
        else:
            from core.domain.docx_transform import export_script_to_docx

        if updated_raw_data is None:
            return {"success": False, "message": "处理raw数据失败：数据为空"}

        if split_mode not in {"auto", "manual"}:
            split_mode = "auto"

        script_data = process_raw_to_script(updated_raw_data, num_segments, split_mode)

        with open(script_path, "w", encoding="utf-8") as handle:
            json.dump(script_data, handle, ensure_ascii=False, indent=2)
        print(f"分段脚本已保存到: {script_path}")

        try:
            export_script_to_docx(script_data, script_docx_path)
            print(f"阅读版DOCX已保存到: {script_docx_path}")
        except Exception as exc:
            print(f"⚠️  生成script.docx失败: {exc}")

        try:
            max_chars_per_line = config.SUBTITLE_MAX_CHARS_PER_LINE
            txt_path = export_plain_text_segments(script_data, paths.text, max_chars_per_line)
            print(f"✅ 纯文本分段文件已保存到: {txt_path}")
        except Exception as exc:
            print(f"⚠️  生成纯文本分段文件失败: {exc}")
            logger.warning(f"生成纯文本分段文件失败: {exc}")

        logger.info(f"步骤1.5处理完成: {script_path}")
        return {
            "success": True,
            "script_data": script_data,
            "script_path": script_path,
            "message": "步骤1.5处理完成",
        }
    except Exception as exc:
        logger.error(f"步骤1.5处理失败: {str(exc)}")
        return {"success": False, "message": f"步骤1.5处理失败: {str(exc)}"}


def run_step_2(
    llm_server: str,
    llm_model: str,
    project_output_dir: str,
    script_path: Optional[str] = None,
    images_method: str = "keywords",
    visual_mode: str = "image",
) -> Dict[str, Any]:
    paths = ProjectPaths(project_output_dir)
    paths.ensure_dirs_exist()
    script_data = load_json_file(script_path) if script_path else load_json_file(paths.script_json())
    if script_data is None:
        return {"success": False, "message": "未找到脚本数据，请先完成步骤1.5"}

    visual_mode = (visual_mode or "image").strip().lower()
    if visual_mode not in getattr(config, "SUPPORTED_VISUAL_MODES", ["image"]):
        return {"success": False, "message": f"不支持的画面模式: {visual_mode}"}

    images_method = images_method or getattr(config, "SUPPORTED_IMAGE_METHODS", ["keywords"])[0]

    raw_data = load_json_file(paths.raw_json()) if os.path.exists(paths.raw_json()) else None
    description_source = _resolve_description_source_text(
        project_output_dir, raw_data=raw_data, script_data=script_data
    )

    if visual_mode in {"youtube", "hybrid"}:
        keywords_data = extract_keywords(llm_server, llm_model, script_data)
        keywords_path = paths.keywords_json()
        _write_json_file(keywords_path, keywords_data)

        description_data = generate_description_summary(
            llm_server, llm_model, description_source or "", max_chars=200
        )
        description_path = paths.mini_summary_json()
        _write_json_file(description_path, description_data)

        media_plan = _build_media_plan(
            llm_server=llm_server,
            llm_model=llm_model,
            script_data=script_data,
            visual_mode=visual_mode,
            keywords_data=keywords_data,
        )
        media_plan_path = paths.media_plan_json()
        _write_json_file(media_plan_path, media_plan)

        return {
            "success": True,
            "keywords_path": keywords_path,
            "mini_summary_path": description_path,
            "media_plan_path": media_plan_path,
            "message": "步骤2完成：已生成关键词、描述小结和可编辑的 media_plan.json",
        }

    if images_method == "description":
        description_data = generate_description_summary(
            llm_server, llm_model, description_source or "", max_chars=200
        )
        description_path = paths.mini_summary_json()
        _write_json_file(description_path, description_data)
        return {"success": True, "mini_summary_path": description_path}

    keywords_data = extract_keywords(llm_server, llm_model, script_data)
    keywords_path = paths.keywords_json()
    _write_json_file(keywords_path, keywords_data)
    return {"success": True, "keywords_path": keywords_path}


def run_step_3(
    image_server: str,
    image_model: str,
    image_size: str,
    image_style_preset: str,
    project_output_dir: str,
    opening_image_style: str,
    images_method: str = "keywords",
    visual_mode: str = "image",
    opening_quote: bool = True,
    target_segments: Optional[List[int]] = None,
    regenerate_opening: bool = True,
    llm_model: Optional[str] = None,
    llm_server: Optional[str] = None,
) -> Dict[str, Any]:
    paths = ProjectPaths(project_output_dir)
    paths.ensure_dirs_exist()

    script_data = load_json_file(paths.script_json())
    if script_data is None:
        return {"success": False, "message": "未找到脚本数据，请先完成步骤1.5"}

    segments = script_data.get("segments", [])
    total_segments = len(segments)
    if total_segments == 0:
        return {"success": False, "message": "脚本中缺少段落内容"}

    visual_mode = (visual_mode or "image").strip().lower()
    if visual_mode not in getattr(config, "SUPPORTED_VISUAL_MODES", ["image"]):
        return {"success": False, "message": f"不支持的画面模式: {visual_mode}"}

    selected_segments: Optional[List[int]] = None
    if target_segments is not None:
        raw_targets = list(target_segments)
        parsed_targets: List[int] = []
        for value in raw_targets:
            try:
                parsed_targets.append(int(value))
            except (TypeError, ValueError):
                continue
        selected_segments = sorted({idx for idx in parsed_targets if 1 <= idx <= total_segments})
        if raw_targets and not selected_segments:
            return {"success": False, "message": f"段落选择无效，请输入 1-{total_segments} 之间的数字"}

    images_method = images_method or getattr(config, "SUPPORTED_IMAGE_METHODS", ["keywords"])[0]
    if llm_model and not llm_server and visual_mode in {"image", "hybrid"}:
        return {"success": False, "message": "步骤3参数错误: 配置了 llm_model 但未配置 llm_server"}

    generation_targets = None if selected_segments is None else selected_segments
    target_set: Set[int] = set(generation_targets or [])

    # image 模式：保持原有行为
    if visual_mode == "image":
        keywords_data = None
        description_data = None
        if images_method == "description":
            description_data = load_json_file(paths.mini_summary_json())
            if description_data is None:
                return {"success": False, "message": "未找到描述小结，请先执行步骤2生成描述"}
        else:
            keywords_data = load_json_file(paths.keywords_json())
            if keywords_data is None:
                return {"success": False, "message": "未找到关键词数据，请先执行步骤2生成关键词"}

        opening_image_path = None
        opening_image_file = paths.opening_image()
        opening_previously_exists = os.path.exists(opening_image_file)
        opening_regenerated = False
        if opening_quote:
            need_refresh = regenerate_opening or not opening_previously_exists
            if need_refresh:
                opening_image_path = generate_opening_image(
                    image_server, image_model, opening_image_style, image_size, paths.media, opening_quote
                )
                opening_regenerated = bool(opening_image_path)
            elif opening_previously_exists:
                opening_image_path = opening_image_file
                print(f"保持现有开场图像: {opening_image_path}")

        if selected_segments is None or len(selected_segments) > 0:
            image_result = generate_images_for_segments(
                image_server,
                image_model,
                script_data,
                image_style_preset,
                image_size,
                paths.media,
                images_method=images_method,
                keywords_data=keywords_data,
                description_data=description_data,
                target_segments=generation_targets,
                llm_model=llm_model,
                llm_server=llm_server,
            )
        else:
            image_paths = []
            for idx in range(1, total_segments + 1):
                segment_path = paths.segment_image(idx)
                image_paths.append(segment_path if os.path.exists(segment_path) else "")
            image_result = {"image_paths": image_paths, "failed_segments": [], "processed_segments": []}

        failed_segments = image_result.get("failed_segments", [])
        if failed_segments:
            failed_str = "、".join(str(idx) for idx in failed_segments)
            return {
                "success": False,
                "message": f"第 {failed_str} 段图像生成失败，请调整提示或稍后重试。",
                "failed_segments": failed_segments,
                "image_paths": image_result.get("image_paths", []),
                "media_paths": image_result.get("image_paths", []),
                "opening_image_path": opening_image_path,
            }

        processed_segments = image_result.get("processed_segments", [])
        if selected_segments is None:
            message = "段落图像生成完成"
            if opening_regenerated:
                message += "，开场图像已更新"
        elif processed_segments:
            seg_text = "、".join(str(idx) for idx in processed_segments)
            message = f"已生成第 {seg_text} 段图像"
            if opening_regenerated:
                message += " 并刷新开场图像"
        else:
            message = "未生成新的段落图像"
            if opening_regenerated:
                message = "已重新生成开场图像"

        payload = {
            "success": True,
            "opening_image_path": opening_image_path,
            "processed_segments": processed_segments,
            "message": message,
            "media_paths": image_result.get("image_paths", []),
            "image_paths": image_result.get("image_paths", []),
        }
        for key, value in image_result.items():
            if key != "processed_segments":
                payload[key] = value
        return payload

    # youtube / hybrid 模式
    media_plan = load_json_file(paths.media_plan_json())
    if not isinstance(media_plan, dict):
        return {"success": False, "message": "缺少 media_plan.json，请先执行步骤2"}

    plan_segments = media_plan.get("segments") or []
    plan_map: Dict[int, Dict[str, Any]] = {}
    for item in plan_segments:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, int):
            plan_map[idx] = item

    cache_dir = os.path.join(paths.media, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    media_paths: List[str] = [""] * total_segments
    for idx in range(1, total_segments + 1):
        existing = _resolve_segment_media_path(paths, idx)
        if existing:
            media_paths[idx - 1] = existing

    opening_media_path = None
    opening_existing = paths.opening_image()
    opening_previously_exists = os.path.exists(opening_existing)
    opening_regenerated = False
    if opening_quote:
        opening_cfg = media_plan.get("opening") if isinstance(media_plan.get("opening"), dict) else {}
        opening_url = str((opening_cfg or {}).get("youtube_url") or "").strip()
        need_refresh = regenerate_opening or not opening_previously_exists
        if need_refresh and is_valid_youtube_url(opening_url):
            try:
                dl = download_youtube_video(opening_url, cache_dir)
                source_path = str(dl.get("local_path") or "")
                source_duration = _safe_float(dl.get("duration"), probe_video_duration(source_path))
                default_opening = _safe_float(getattr(config, "YOUTUBE_OPENING_DEFAULT_DURATION_SECONDS", 6.5), 6.5)
                start, end = _clip_interval(
                    opening_cfg.get("start_seconds", 0.0),
                    opening_cfg.get("end_seconds", default_opening),
                    candidate_duration=source_duration,
                    target_duration=default_opening,
                )
                opening_media_path = os.path.join(paths.media, "opening.mp4")
                trim_video_clip(source_path, opening_media_path, start, end)
                opening_regenerated = True
            except Exception as exc:
                logger.warning(f"开场YouTube素材生成失败: {exc}")
        elif opening_previously_exists:
            opening_media_path = opening_existing

    candidate_indices = selected_segments if selected_segments is not None else list(range(1, total_segments + 1))
    youtube_targets: List[int] = []
    image_targets: List[int] = []
    for idx in candidate_indices:
        plan_item = plan_map.get(idx, {})
        source = str(plan_item.get("source") or "image").strip().lower()
        if visual_mode == "youtube":
            source = "youtube"
        if source == "youtube":
            youtube_targets.append(idx)
        else:
            image_targets.append(idx)

    failed_segments: List[int] = []
    processed_segments: List[int] = []

    for idx in youtube_targets:
        plan_item = plan_map.get(idx, {})
        youtube_url = str(plan_item.get("youtube_url") or "").strip()
        if not is_valid_youtube_url(youtube_url):
            failed_segments.append(idx)
            continue

        try:
            dl = download_youtube_video(youtube_url, cache_dir)
            source_path = str(dl.get("local_path") or "")
            source_duration = _safe_float(dl.get("duration"), probe_video_duration(source_path))
            default_clip = _safe_float(getattr(config, "YOUTUBE_DEFAULT_CLIP_DURATION_SECONDS", 6.5), 6.5)
            target_duration = _safe_float(plan_item.get("duration_seconds", default_clip), default_clip)
            start, end = _clip_interval(
                plan_item.get("start_seconds", 0.0),
                plan_item.get("end_seconds", target_duration),
                candidate_duration=source_duration,
                target_duration=target_duration,
            )
            output_path = os.path.join(paths.media, f"segment_{idx}.mp4")
            trim_video_clip(source_path, output_path, start, end)
            media_paths[idx - 1] = output_path
            processed_segments.append(idx)
        except Exception as exc:
            logger.warning(f"第{idx}段YouTube素材生成失败: {exc}")
            failed_segments.append(idx)

    if image_targets:
        keywords_data = None
        description_data = None
        if images_method == "description":
            description_data = load_json_file(paths.mini_summary_json())
            if description_data is None:
                return {"success": False, "message": "未找到描述小结，请先执行步骤2生成描述"}
        else:
            keywords_data = load_json_file(paths.keywords_json())
            if keywords_data is None:
                return {"success": False, "message": "未找到关键词数据，请先执行步骤2生成关键词"}

        image_result = generate_images_for_segments(
            image_server,
            image_model,
            script_data,
            image_style_preset,
            image_size,
            paths.media,
            images_method=images_method,
            keywords_data=keywords_data,
            description_data=description_data,
            target_segments=image_targets,
            llm_model=llm_model,
            llm_server=llm_server,
        )
        for idx, media_path in enumerate(image_result.get("image_paths", []), 1):
            if media_path:
                media_paths[idx - 1] = media_path
        processed_segments.extend(image_result.get("processed_segments", []))
        failed_segments.extend(image_result.get("failed_segments", []))

    failed_segments = sorted(set(failed_segments))
    processed_segments = sorted(set(processed_segments))
    if failed_segments:
        failed_str = "、".join(str(idx) for idx in failed_segments)
        return {
            "success": False,
            "message": f"第 {failed_str} 段素材生成失败，请检查 media_plan 或稍后重试。",
            "failed_segments": failed_segments,
            "media_paths": media_paths,
            "image_paths": media_paths,
            "opening_image_path": opening_media_path,
        }

    if selected_segments is None:
        message = "段落素材生成完成"
    elif processed_segments:
        seg_text = "、".join(str(idx) for idx in processed_segments)
        message = f"已生成第 {seg_text} 段素材"
    else:
        message = "未生成新的段落素材"
    if opening_regenerated:
        message += "，开场素材已更新"

    return {
        "success": True,
        "opening_image_path": opening_media_path,
        "processed_segments": processed_segments,
        "media_paths": media_paths,
        "image_paths": media_paths,
        "message": message,
    }


def run_step_4(
    tts_server: str,
    voice: str,
    project_output_dir: str,
    opening_quote: bool = True,
    target_segments: Optional[List[int]] = None,
    regenerate_opening: bool = True,
    speech_rate: int = 0,
    loudness_rate: int = 0,
    emotion: str = "neutral",
    emotion_scale: int = 4,
    mute_cut_threshold: int = 400,
    mute_cut_min_silence_ms: int = 200,
    mute_cut_remain_ms: int = 100,
) -> Dict[str, Any]:
    paths = ProjectPaths(project_output_dir)
    paths.ensure_dirs_exist()

    script_data = load_json_file(paths.script_json())
    if script_data is None:
        return {"success": False, "message": "未找到脚本数据，请先完成步骤1.5"}

    segments = script_data.get("segments", [])
    total_segments = len(segments)
    if total_segments == 0:
        return {"success": False, "message": "脚本中缺少段落内容"}

    selected_segments: Optional[List[int]] = None
    if target_segments is not None:
        raw_targets = list(target_segments)
        parsed_targets: List[int] = []
        for value in raw_targets:
            try:
                parsed_targets.append(int(value))
            except (TypeError, ValueError):
                continue
        selected_segments = sorted({idx for idx in parsed_targets if 1 <= idx <= total_segments})
        if raw_targets and not selected_segments:
            return {"success": False, "message": f"段落选择无效，请输入 1-{total_segments} 之间的数字"}

    generation_targets = None if selected_segments is None else selected_segments
    voice_result = synthesize_voice_for_segments(
        tts_server,
        voice,
        script_data,
        paths.voice,
        target_segments=generation_targets,
        speech_rate=speech_rate,
        loudness_rate=loudness_rate,
        emotion=emotion,
        emotion_scale=emotion_scale,
        mute_cut_threshold=mute_cut_threshold,
        mute_cut_min_silence_ms=mute_cut_min_silence_ms,
        mute_cut_remain_ms=mute_cut_remain_ms,
    )
    audio_paths = voice_result.get("audio_paths", [])
    missing_segments = voice_result.get("missing_segments", [])

    opening_audio_file = paths.opening_audio()
    opening_previously_exists = os.path.exists(opening_audio_file)
    narration_path = _invoke_opening_narration(
        script_data,
        paths.voice,
        voice,
        opening_quote,
        announce=True,
        force_regenerate=regenerate_opening,
        speech_rate=speech_rate,
        loudness_rate=loudness_rate,
        emotion=emotion,
        emotion_scale=emotion_scale,
        mute_cut_threshold=mute_cut_threshold,
        mute_cut_min_silence_ms=mute_cut_min_silence_ms,
        mute_cut_remain_ms=mute_cut_remain_ms,
    )

    opening_refreshed = bool(opening_quote and narration_path and (regenerate_opening or not opening_previously_exists))
    processed_segments = list(range(1, total_segments + 1)) if selected_segments is None else list(selected_segments)

    if selected_segments is None:
        message = "段落语音生成完成"
        if opening_refreshed:
            message += "，开场金句音频已更新"
    elif processed_segments:
        seg_text = "、".join(str(idx) for idx in processed_segments)
        message = f"已生成第 {seg_text} 段语音"
        if opening_refreshed:
            message += " 并刷新开场金句音频"
    else:
        message = "未生成新的段落语音"
        if opening_refreshed:
            message = "已重新生成开场金句音频"

    return {
        "success": True,
        "audio_paths": audio_paths,
        "processed_segments": processed_segments,
        "missing_segments": missing_segments,
        "message": message,
    }


def run_step_5(
    project_output_dir: str,
    image_size: str,
    enable_subtitles: bool,
    bgm_filename: str,
    voice: str,
    opening_quote: bool = True,
    speech_rate: int = 0,
    loudness_rate: int = 0,
    emotion: str = "neutral",
    emotion_scale: int = 4,
    mute_cut_threshold: int = 400,
    mute_cut_min_silence_ms: int = 200,
    mute_cut_remain_ms: int = 100,
) -> Dict[str, Any]:
    project_root = _get_project_root()
    paths = ProjectPaths(project_output_dir)

    if not os.path.exists(paths.script_json()):
        return {"success": False, "message": "脚本文件不存在，请先完成步骤1.5"}

    script_data = load_json_file(paths.script_json())
    if not script_data:
        return {"success": False, "message": "脚本文件加载失败"}

    expected_segments = script_data.get("actual_segments", 0)
    image_paths = []
    for idx in range(1, expected_segments + 1):
        media_path = _resolve_segment_media_path(paths, idx)
        if media_path:
            image_paths.append(media_path)
    image_count = len(image_paths)

    audio_count = 0
    for idx in range(1, expected_segments + 1):
        if paths.segment_audio_exists(idx):
            audio_count += 1

    if image_count == 0:
        return {"success": False, "message": "未找到图像文件，请先完成步骤3"}
    if audio_count == 0:
        return {"success": False, "message": "未找到音频文件，请先完成步骤4"}
    if image_count != expected_segments:
        return {"success": False, "message": f"图像文件不完整，需要{expected_segments}个，找到{image_count}个"}
    if audio_count != expected_segments:
        return {"success": False, "message": f"音频文件不完整，需要{expected_segments}个，找到{audio_count}个"}

    audio_paths = []
    for idx in range(1, script_data.get("actual_segments", 0) + 1):
        audio_path = paths.segment_audio_exists(idx)
        if audio_path:
            audio_paths.append(audio_path)

    bgm_audio_path = _resolve_bgm_audio_path(bgm_filename, project_root)
    opening_image_candidate = paths.opening_image() if os.path.exists(paths.opening_image()) else None
    opening_golden_quote = (script_data or {}).get("golden_quote", "")
    opening_narration_audio_path = _invoke_opening_narration(
        script_data,
        paths.voice,
        voice,
        opening_quote,
        speech_rate=speech_rate,
        loudness_rate=loudness_rate,
        emotion=emotion,
        emotion_scale=emotion_scale,
        mute_cut_threshold=mute_cut_threshold,
        mute_cut_min_silence_ms=mute_cut_min_silence_ms,
        mute_cut_remain_ms=mute_cut_remain_ms,
    )

    composer = VideoComposer()
    final_video_path = composer.compose_video(
        image_paths,
        audio_paths,
        paths.final_video(),
        script_data=script_data,
        enable_subtitles=enable_subtitles,
        bgm_audio_path=bgm_audio_path,
        opening_image_path=opening_image_candidate,
        opening_golden_quote=opening_golden_quote,
        opening_narration_audio_path=opening_narration_audio_path,
        bgm_volume=float(getattr(config, "BGM_DEFAULT_VOLUME", 0.2)),
        narration_volume=float(getattr(config, "NARRATION_DEFAULT_VOLUME", 1.0)),
        image_size=image_size,
        opening_quote=opening_quote,
        project_root=project_output_dir,
    )

    return {"success": True, "final_video": final_video_path}


def _run_cover_generation(
    project_output_dir: str,
    cover_image_size: Optional[str],
    cover_image_server: str,
    cover_image_model: Optional[str],
    cover_image_style: Optional[str],
    cover_image_count: int,
    script_data: Optional[Dict[str, Any]],
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not script_data and not raw_data:
        raise ValueError("缺少脚本或原始数据")

    base = script_data or raw_data or {}
    video_title = base.get("title") or "未命名视频"
    content_title = base.get("content_title") or video_title
    cover_subtitle = base.get("cover_subtitle") or ""

    if cover_image_size:
        from core.config import COVER_IMAGE_SIZE_PRESETS

        if cover_image_size in COVER_IMAGE_SIZE_PRESETS:
            cover_image_size = COVER_IMAGE_SIZE_PRESETS[cover_image_size]
    else:
        from core.config import COVER_IMAGE_SIZE_PRESETS, DEFAULT_COVER_IMAGE_SIZE_KEY

        cover_image_size = COVER_IMAGE_SIZE_PRESETS.get(DEFAULT_COVER_IMAGE_SIZE_KEY, "2048x2048")

    cover_image_model = cover_image_model or config.RECOMMENDED_MODELS["image"].get(
        cover_image_server, ["doubao-seedream-4-0-250828"]
    )[0]
    cover_image_style = cover_image_style or "cover01"
    if not cover_image_server:
        raise ValueError("封面参数错误: cover_image_server 不能为空")
    try:
        Config.validate_parameters(
            target_length=config.MIN_TARGET_LENGTH,
            num_segments=config.MIN_NUM_SEGMENTS,
            llm_server=config.SUPPORTED_LLM_SERVERS[0],
            image_server=cover_image_server,
            tts_server=config.SUPPORTED_TTS_SERVERS[0],
            image_model=cover_image_model,
            image_size=cover_image_size,
            llm_model=config.RECOMMENDED_MODELS["llm"][config.SUPPORTED_LLM_SERVERS[0]][0],
        )
    except Exception as exc:
        raise ValueError(f"封面参数校验失败: {exc}") from exc

    return generate_cover_images(
        project_output_dir,
        cover_image_server,
        cover_image_model,
        cover_image_size,
        cover_image_style,
        max(1, int(cover_image_count or 1)),
        video_title,
        content_title,
        cover_subtitle,
    )


def run_step_6(
    project_output_dir: str,
    cover_image_size: str,
    cover_image_server: Optional[str] = None,
    cover_image_model: Optional[str] = None,
    cover_image_style: str = "cover01",
    cover_image_count: int = 1,
) -> Dict[str, Any]:
    paths = ProjectPaths(project_output_dir)
    if not os.path.exists(paths.raw_json()):
        return {"success": False, "message": "缺少 raw.json，请先完成步骤1"}

    raw_data = load_json_file(paths.raw_json())
    if raw_data is None:
        return {"success": False, "message": "raw.json 加载失败"}

    script_data = load_json_file(paths.script_json()) if os.path.exists(paths.script_json()) else None

    try:
        cover_image_server = cover_image_server or getattr(config, "COVER_IMAGE_SERVER", "")
        cover_result = _run_cover_generation(
            project_output_dir,
            cover_image_size,
            cover_image_server,
            cover_image_model,
            cover_image_style,
            cover_image_count,
            script_data,
            raw_data,
        )
        return {"success": True, **cover_result}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


__all__ = [
    "run_step_1",
    "run_step_1_5",
    "run_step_2",
    "run_step_3",
    "run_step_4",
    "run_step_5",
    "run_step_6",
    "_get_project_root",
    "_initialize_project",
    "_resolve_bgm_audio_path",
    "_ensure_opening_narration",
    "_invoke_opening_narration",
    "_resolve_description_source_text",
    "_run_cover_generation",
]

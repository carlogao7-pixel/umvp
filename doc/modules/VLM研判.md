# VLM 研判 VLMAnalyzer

> 模块组：管道模块 | 代码：`umvp/pipe/composer.py`（`VLMAnalyzer`）
> 数据库参数：`VLM0001`「警用无人机·VLM研判」（presets 表 + preset_params_vlm 表）

## 1. 模块定位

大模型研判：把 YOLO 的报送请求加工成 VLM 素材包（裁剪/缩放/提示词），
调用视觉大模型（如 qwen3-vl-32b）对画面做语义研判。模块自身只做
**素材准备**与**结果回调封装**，真实推理通过外部端点调用（OpenAI 兼容接口）。

## 2. 参数

### 2.1 模块自身参数（落库为 preset）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| max_resolution | str | "1280,720" | 送审前缩放上限（宽,高），超出按比例缩小 |
| prompt | text | 空 | VLM 提示词；空=用测试台默认分类提示词 |

### 2.2 VLM 接入参数（web_lab 链路运行器读取，落库在同一 preset）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| use_real_vlm | bool | false | true=真发 HTTP 请求到端点；false=只演示素材包不推理 |
| vlm_endpoint | str | （见库内值） | OpenAI 兼容接口地址（/v1/chat/completions） |
| vlm_model | str | qwen3-vl-32b | 调用的模型名 |
| vlm_key | str | 空 | API Key，免鉴权端点留空 |

### 2.3 单模块测试参数（仅模块测试页运行用，不落库、不进链路）

`image`（真实调用取材图片）、`ref`（素材引用）、`gran`（报送粒度）、`track_key`、
`bbox`、`det_count`——构造单次演示素材包的回填字段，链路运行时由真实
SubmitRequest 提供，不预填、不落库。

### 2.4 当前库内值（VLM0001）

`max_resolution="1280,720", use_real_vlm=true,
vlm_endpoint=http://117.42.21.253:8000/v1/chat/completions,
vlm_model=qwen3-vl-32b`，prompt 为警情四分类提示词（要求返回 JSON：
`alert_type`（治安类/交警类/群体性事件类/救援救助类/无异常）+ `description`）。

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| req | SubmitRequest | YOLO 的报送请求（ref/gran/track_key/dets 等，见《YOLO识别》§4.2） |
| frame | ndarray (BGR) | 报送时刻的原始帧（裁剪/缩放取材用） |
| infer | 回调 | 推理注入：`analyze(payload, infer)`，真实链路为 HTTP 调用封装 |

## 4. 输出（字段和内容）

### 4.1 prepare() → payload（素材包 dict）

| 字段 | 类型 | 内容 |
|---|---|---|
| ref | str | 素材引用（full:cls{N} / full:global）；ROI 裁剪送审改由上游「目标裁剪」阶段承担 |
| gran | str | 报送粒度（class/scene） |
| track_key | int | 识别类型编号（scene=-1） |
| prompt | str | 提示词 |
| max_resolution | (w,h) | 送审缩放上限 |
| bbox | (x1,y1,x2,y2)/None | 单目标位置（有单目标素材时） |
| det_count | int | 该类目标数量 |

### 4.2 链路运行时的 VLM 研判结果

真实 VLM 返回 JSON 文本，链路解析出两个字段喂给输出处理（OutputPolicy）：

| 字段 | 类型 | 内容 |
|---|---|---|
| alert_type（action） | str | 警情类型/动作名（如"交警类"、"无异常"），输出处理按它匹配 target_actions |
| description | str | 警情文字描述，随 AlarmEvent 带出 |

## 5. 上下游衔接

- 上游：YOLO 识别的 SubmitRequest + 原帧。
- 下游：研判结果（action, description）交给输出处理 `on_vlm_result()`（兼容入口，内部走统一漏斗）判定告警。

## 6. 验证

web_lab「大模型研判 VLMAnalyzer」单模块测试（可勾选 use_real_vlm 实测端点）；
链路级验证见《警用无人机链路》§6（vlm_feedback 开关打开即真实调用）。

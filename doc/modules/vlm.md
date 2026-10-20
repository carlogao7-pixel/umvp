# VLM 研判 VLMAnalyzer

> 模块组：管道通用模块 | 代码：`umvp/pipe/composer.py`（`VLMAnalyzer`）
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
| image_source | select | annotated | **送审图像源**（连线相关，2026-10-16 新增）：`annotated`＝带框图（缺则回退原帧，＝旧行为）/ `frame`＝本帧原帧 / `crops`＝与送审目标匹配的裁剪子图（按 bbox 精确或 IoU≥0.5 命中，**链路须含 `target_crop` 阶段**；未命中回退并在 timeline 标注素材来源）。缺省＝现状（零回归），缺源一律回退不报错 |
| image_pick | select | first | **送审张数**（2026-10-16 新增）：`first`＝只送首张（＝旧行为，逐字等价）/ `all`＝送**图集全部**（`image_source=crops` 时该类全部命中对象子图，OpenAI 兼容**多图** payload）。与 `image_source` 正交：前者选"哪张"、后者选"几张" |
| use_real_vlm | bool | false | true=真发 HTTP 请求到端点；false=只演示素材包不推理 |
| vlm_endpoint | str | （见库内值） | OpenAI 兼容接口地址（/v1/chat/completions）；**环境变量 `VLM_ENDPOINT` 优先** |
| vlm_model | str | qwen3-vl-32b | 调用的模型名；**环境变量 `VLM_MODEL` 优先** |
| vlm_key | str | 空 | API Key，免鉴权端点留空；**环境变量 `VLM_API_KEY`（兼容 `GLM_API_KEY`）优先** |

> **端点/模型/key 三级优先级**（`web_lab/server.py:_resolve_vlm_cfg`，2026-10-19 WO-0006）：
> **环境变量 > 链路 spec 阶段参数 > 内置默认**。环境变量 `VLM_ENDPOINT` / `VLM_MODEL` /
> `VLM_API_KEY`（key 兼容别名 `GLM_API_KEY`，两者都设时 `VLM_API_KEY` 优先）为**部署级强制覆盖**——
> 导出即全链路切到该端点、无需改 DB；**不设任何 env 时逐字回落到改动前行为（零回归）**。
> 端点返回 401/403（鉴权）/429（限额）时抛 `VlmHttpError`（携带中文可读提示 `hint`），
> 驱动层每轮**最多提示一次**（参见 [警用无人机链路](../chains/pipe0001_police_uav.md) §5 失败处理）。
> 环境变量注入只存在于驱动层（`server.py`），模块自身（`VLMAnalyzer`）不读环境变量。

> **多图送审实测**：`image_pick=all` 的 OpenAI 兼容多图 payload 已由真实端点（`glm-4v-flash`）
> 实证支持（见收件箱 INB-0040）；单图路径（`first`，默认）与改动前逐字等价。

> **连线（inputs）**：`vlm.inputs` 仅支持 `dets` 端口（决定"报送哪些检测"，如 `yolo.dets`＝筛选前 / `det_filter.dets`＝筛选后）；`vlm.inputs.image` **不支持**——图片源请用 `image_source`。

> `validate_chain` 对 `image_source` 非法取值报错；`annotated` 但上游无 `annotate`、`crops` 但上游无
> `target_crop` 时给**非致命警告**（运行时回退）。详见
> [拼接层端口协议设计稿](../design/assembly_ports.md)（P5，第 1 步已实施）。

### 2.3 单模块测试参数（仅模块测试页运行用，不落库、不进链路）

`image`（真实调用取材图片）、`ref`（素材引用）、`gran`（报送粒度）、`track_key`、
`bbox`、`det_count`——构造单次演示素材包的回填字段，链路运行时由真实
SubmitRequest 提供，不预填、不落库。

### 2.4 当前库内值（VLM0001，快照 2026-10）

`max_resolution="1280,720", use_real_vlm=true,
vlm_endpoint=http://117.42.21.253:8000/v1/chat/completions,
vlm_model=qwen3-vl-32b`，prompt 为警情四分类提示词（要求返回 JSON：
`alert_type`（治安类/交警类/群体性事件类/救援救助类/无异常）+ `description`）。

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| req | SubmitRequest | YOLO 的报送请求（ref/gran/track_key/dets 等，见 [YOLO 识别](yolo.md) §4） |
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

真实 VLM 返回 JSON 文本，链路解析出字段喂给事件尾（SeqFilter）：

| 字段 | 类型 | 内容 |
|---|---|---|
| alert_type（action） | str | 警情类型/动作名（如"交警类"、"无异常"），事件尾按它匹配规则 `match`（词表）并作动态事件名 |
| description | str | 警情文字描述，随记录 `content`/事件 detail 带出 |

## 5. 上下游衔接

- 上游：YOLO 识别的 SubmitRequest + 原帧。
- 下游：研判记录经 `Pipeline.publish()` 走事件尾（`event_filter`）——词表命中+冷却后出事件。
- **端口声明（P5）**：`VLMAnalyzer.INPUTS = {"image": "Image", "dets": "Records"}`、`OUTPUTS = {}`
  ——模块只做素材准备/回调封装，**不产出登记产物**（研判内容经 `publish` 回写记录的
  `label`/`content` 走事件尾，不在 `Pipeline.step` 内登记为产物）。
- **连线（inputs）**：只有 `vlm.inputs.dets` 由驱动层解析（`_apply_vlm_inputs` 按声明从
  `pipe.artifacts` 取检测记录重构送审请求，如 `yolo.dets`＝筛选前 / `det_filter.dets`＝筛选后）；
  `vlm.inputs.image` **不接受**（图片源统一由 `image_source` 参数管，避免双机制，校验器报错并提示）。
  未声明 `inputs` 的旧 spec 不报错不警告（零回归）。
- **送审图像源与张数**：`image_source`（annotated/frame/crops）选"哪张"、`image_pick`（first/all）选"几张"，
  由驱动层 `_pick_vlm_images` 按 `records.image_set_of`（集合语义）取图。

## 6. 验证

web_lab「大模型研判 VLMAnalyzer」单模块测试（可勾选 use_real_vlm 实测端点）；
链路级验证见 [警用无人机链路](../chains/pipe0001_police_uav.md) §6（vlm_feedback 开关打开即真实调用）。

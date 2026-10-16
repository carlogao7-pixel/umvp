# 模块文档索引（doc/modules/）

> 本目录每模块一篇，**按「记录数据流」角色分组**（不按业务域/链路）。
> **模块篇**体例：模块定位 → 参数 → 输入 → 输出 → 上下游衔接 → 验证（可选附录：调参速查 / 注意事项 / 边界与路线）。
> 参数以本目录为准；三条保留链路专篇见 [`../chains/`](../chains/)；项目总览见 [`doc/README.md`](../README.md)。

## 〇、通用语（先读）

模块间通用语 = **dict 记录为 JSON 主干** ＋ **图像与向量走旁路字段**（契约见
[`umvp/pipe/records.py`](../../umvp/pipe/records.py)）：

- **JSON 主干**：记录必备 `kind` / `ts` / `frame_idx`；通用字段 `bbox` / `score` / `cls_id` /
  `track_id` / `label` / `text`；事件字段 `events`（列表，一条记录可携带多个事件）。
- **旁路字段**（约定不序列化）：`image`（ndarray 子图/帧，**记录旁路图像统一键名**）、`embedding`（向量）——像素与向量不进 JSON。
  历史模块自造键为**兼容别名**（`to_record` 转换时自动补 `image`，旧键保留）：
  目标裁剪 `img` / 人脸检测 `face_img` / 车牌识别 `plate_img`。
- **适配与事件 API**：老对象在边界经 `to_record` 统一（dataclass 自动转，特殊类型 `@register_adapter`）；
  `add_event` / `get_events` / `collect_events` 为通用事件接口；派生字段 `w/h/short/long/area`
  由 [`umvp/pipe/conditions.py`](../../umvp/pipe/conditions.py) 提供。

### 产物端口表（模块｜端口与产物｜接产改；2026-10-16 端口协议）

> 模块间连线的通用语（P5 端口协议，见 [拼接层端口协议设计稿](../design/assembly_ports.md)）：
> **产物名 = `{阶段}.{产物}`**，类型三类 `Image / Records / Vector`；模块以类属性
> `INPUTS` / `OUTPUTS` 自描述（如 `Annotator.INPUTS == {"image": "Image", "dets": "Records"}`）；
> spec 阶段可用 `"inputs": {端口: 上游产物名}` 显式连线（**不写 = 现状默认**）。

| 模块 | INPUTS（端口→类型） | OUTPUTS（产物→类型） | 备注（接/产/改） |
|---|---|---|---|
| [帧管理](frame_manager.md) | `{}` | `frame_manager.frame` → Image | 产原图（帧元信息进主干） |
| [YOLO 识别](yolo.md) | `image` → Image | `yolo.dets` → Records | 纯推理产检测记录 |
| [检测筛选](det_filter.md) | `dets` → Records | `det_filter.dets` → Records | 放行子集＋`events`（改） |
| [时序判定](seq_filter.md) | `dets` → Records | `seq_filter.dets` → Records | 门控放行＋`events`（改） |
| [目标裁剪](target_crop.md) | `frame` → Image；`dets` → Records | `target_crop.crops` → Records（元素含 `image`） | 接检测产子图 |
| [标注](annotate.md) | `image` → Image；`dets` → Records | `annotate.annotated` → Image | 产带框图 |
| [VLM 研判](vlm.md) | `image` → Image；`dets` → Records | —（研判内容回写主干 `label`/`content`） | 送审在驱动层消费 |
| [人脸检测](face_detect.md) | `frame` → Image；`crops` → Records | `face_detect.faces` → Records（元素含 `image`） | 接裁剪产人脸记录＋人脸图 |
| [车牌识别](plate_recog.md) | `frame` → Image；`crops` → Records | `plate_recog.plates` → Records（元素含 `image`） | 接裁剪产车牌记录＋原分辨率车牌图 |
| [人脸嵌入](face_embed.md) | 人脸图 | 512d 向量 → Vector | 向量旁路 |
| [向量底库](face_store.md) | 查询向量 | MatchResult 记录 | 检索产出 |

**本期连线范围**：`inputs` 解析已覆盖 `yolo.image`、`det_filter.dets`、`seq_filter.dets`、
`target_crop.frame/dets`、`annotate.image/dets`；`vlm` / `plate_recog` / `face_detect` 暂按默认取数
（写了 `inputs` 会由 `validate_chain` 给出**非致命警告**）。校验：未知产物 / 非法端口 / 非上游引用 /
类型不符 / 值非字符串 → 报错；同一产物被多个下游引用 → 允许（分支留路）。

## 一、主干生产者（向数据流放入新记录）

> 什么时候读：链路最上游——帧怎么进来、目标怎么被检出。

| 文档 | 模块 | 代码 | 主要 preset |
|---|---|---|---|
| [帧管理](frame_manager.md) | FrameManager 取帧节奏 + 背压（帧记录与原图旁路的起点） | `FrameManager` | FM0001~0003 |
| [YOLO 识别](yolo.md) | 小模型识别（**纯推理**，产 Detection 记录） | `YOLODetector` | YOLO0001~0003 |

## 二、旁路加工与识别（消费图像旁路 → 产新记录或改写主干）

> 什么时候读：要裁子图、画框、送大模型、从子图里做二次识别（检脸/检牌）时。

| 文档 | 模块 | 代码 | 主要 preset |
|---|---|---|---|
| [目标裁剪](target_crop.md) | 按上游检测框裁子图（产 CropResult + 图像旁路 `image`，旧名 `img`） | `TargetCrop` / `crop_targets` | CROP0001+ |
| [标注](annotate.md) | 帧副本画框（图像旁路，供 VLM 看"原图+画框"） | `Annotator` / `annotate_frame` | ANNO0001+ |
| [VLM 研判](vlm.md) | 大模型研判 + 素材准备（研判内容回写主干） | `VLMAnalyzer` | VLM0001 |
| [人脸检测](face_detect.md) | SCRFD 检测 + 5 点对齐；`FaceStage` 子图检脸 + 原图出脸 | `FaceDetector` / `FaceStage` | FDET0001~0002 |
| [车牌识别](plate_recog.md) | HyperLPR3 识别；`PlateStage` 车图检牌 + 原图出牌 | `PlateRecognizer` / `extract_plates` / `PlateStage` | PLATE0001~0002 |

## 三、主干筛选与事件（只取舍 / 标注，不改记录内容）

> 什么时候读：要定有效性、跨帧确认、给幸存记录追加事件（作事件尾）时。

| 文档 | 模块 | 代码 | 主要 preset |
|---|---|---|---|
| [检测筛选](det_filter.md) | 有效性筛选（逐条条件 DSL + 聚合 count + 成组数量 + 事件标注） | `DetFilter` | DF0001+ |
| [时序判定](seq_filter.md) | 时序门控（节流/变化）+ 窗口确认闸 + 事件尾 | `SeqFilter` / `SeqRule` | SEQ0001+ |

> 链路输出 = 链尾阶段产出的事件：以幸存记录携带的 `events` 为载体
> （[检测筛选](det_filter.md) `events` 规格 / [时序判定](seq_filter.md) `rules`），
> 不再有"另造告警对象"的物化阶段。链路终段允许 `det_filter`/`seq_filter` 收尾。

## 四、向量旁路（人脸图 → 向量 → 检索结论）

> 什么时候读：做"人脸向量注册 / 检索"（截取之外的识别链路）时。

| 文档 | 模块 | 代码 | 主要 preset |
|---|---|---|---|
| [人脸嵌入](face_embed.md) | ArcFace 512d 嵌入（人脸图 → 向量旁路） | `FaceEmbedder` | FEMB0001 |
| [向量底库](face_store.md) | numpy 线性检索底库（向量 → MatchResult 记录） | `FaceStore` | FSTO0001 |

## 五、通用契约与工具（跨链路共用，无独立 preset）

> 什么时候读：约定记录格式 / 写筛选条件 / 做坐标还原裁剪时（被多个模块复用）。

| 工具 | 代码 | 说明 |
|---|---|---|
| Record 记录适配 | `umvp/pipe/records.py` | 模块间通用记录约定（dict 通用语，图像/向量走旁路字段）+ `to_record` 对象适配 + 通用事件 API |
| 条件 DSL 引擎 | `umvp/pipe/conditions.py` | 框/记录筛选通用语法（与 `,` / 或 `;` / 非 `not` / 括号嵌套 + 派生字段 w/h/short/long/area + 聚合 count）；检测筛选、目标裁剪、标注共用 |
| `Cropper` 通用裁剪 | `umvp/pipe/cropper.py` | 按框裁图可缩放 / 坐标还原 / 原分辨率提取；人脸、车牌链共用（旧名 `CropRestore`/`PersonCrop` 为兼容别名） |

## 保留链路

三条保留链路专篇见 [`../chains/`](../chains/)：PIPE0001 警用无人机 /
PIPE0002 车牌识别 / PIPE0003 人脸截取。

## 相关文档

- [`doc/README.md`](../README.md) —— 文档总览（业务架构 / 技术结构 / 文件索引）

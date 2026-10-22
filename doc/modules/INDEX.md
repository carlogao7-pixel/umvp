# 模块文档索引（doc/modules/）

> 本目录每模块一篇，**按「记录数据流」角色分组**（不按业务域/链路）。
> **模块篇**体例：模块定位 → 参数 → 输入 → 输出 → 上下游衔接 → 验证（可选附录：调参速查 / 注意事项 / 边界与路线）。
> 参数以本目录为准；三条保留链路专篇见 [`../chains/`](../chains/)；项目总览见 [`doc/README.md`](../README.md)。

## 〇、通用语（先读）

模块间通用语 = **dict 记录为 JSON 主干** ＋ **图像与向量走旁路字段**（契约见
[`umvp/pipe/records.py`](../../umvp/pipe/records.py)）：

- **JSON 主干**：记录必备 `kind` / `ts` / `frame_idx`；通用字段 `bbox` / `score` / `cls_id` /
  `track_id` / `label` / `text`；事件字段 `events`（列表，一条记录可携带多个事件）。
  检测产物（`yolo.dets` 起）为**原生 dict 记录**（`kind="det"`；`ts`/`frame_idx` 由
  `Pipeline.step` 注入）；`Detection` dataclass 仅保留供外部构造/直传，直传入口经
  `to_record` 归一为 dict。
- **旁路字段**（约定不序列化）：`image`（ndarray 子图/帧，**记录旁路图像统一键名**；
  **或 ndarray 列表＝图集**——一条记录需多图时以集合形式携带、**不拆记录**）、`embedding`（向量）——像素与向量不进 JSON。
  读取器两个（语义有别）：**`image_of(rec)`＝单图语义**（图集时取首张，兼容既有调用）、
  **`image_set_of(rec)`＝集合语义**（单图→长度 1 列表、图集→原样、无→空）；下游要消费
  "整条记录的图片集合"用后者。
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
| [YOLO 识别](yolo.md) | `image` → Image | `yolo.dets` → Records | 纯推理产检测记录（dict，`kind=det`） |
| [检测筛选](det_filter.md) | `dets` → Records | `det_filter.dets` → Records | 放行子集＋`events`（改） |
| [时序判定](seq_filter.md) | `dets` → Records | `seq_filter.dets` → Records | 门控放行＋`events`（改） |
| [目标裁剪](target_crop.md) | `frame` → Image；`dets` → Records | `target_crop.crops` → Records（元素含 `image`） | 接检测产子图 |
| [标注](annotate.md) | `image` → Image；`dets` → Records | `annotate.annotated` → Image | 产带框图 |
| [VLM 研判](vlm.md) | `image` → Image；`dets` → Records | —（研判内容回写主干 `label`/`content`） | 送审在驱动层消费 |
| [人脸检测](face_detect.md) | `frame` → Image；`crops` → Records | `face_detect.faces` → Records（元素含 `image`） | 接裁剪产人脸记录＋人脸图 |
| [车牌识别](plate_recog.md) | `frame` → Image；`crops` → Records | `plate_recog.plates` → Records（元素含 `image`） | 接裁剪产车牌记录＋原分辨率车牌图 |
| [人脸嵌入](face_embed.md) | 人脸图 | 512d 向量 → Vector | 向量旁路 |
| [向量底库](face_store.md) | 查询向量 | MatchResult 记录 | 检索产出 |

> 上表前 9 行为**已声明 `INPUTS`/`OUTPUTS` 的逐帧阶段**（端口/产物名即代码里的类属性，权威来源）；
> 末两行（人脸嵌入 / 向量底库）为向量旁路模块，**未声明端口、不登记产物**，表中仅为语义说明。

**基数约定（2026-10-16）**：记录数不随图片数膨胀——识别模块一条记录可含多个识别对象；图像生产
模块**一条记录需多图时，以「图集」（ndarray 列表）放进 `image` 旁路，不拆记录**；下游自行处理
单图/图集（标注：一条记录 → 一张图多框，不受影响）。注意区分：**目标裁剪是"逐对象一条记录"
（每个对象一条、各带一张 `image`），不属"一条记录多图"**；图集只用于"一条记录确需多图"的场景
（如按类报送时该类全部命中子图），权威约定见
[拼接层端口协议设计稿](../design/assembly_ports.md) §3⑥。

**本期连线范围**（唯一来源 `web_lab/server.py` 的 `CHAIN_RESOLVED_PORTS` ＝
`pipe.composer.RESOLVED_PORTS` ＋ 驱动层额外解析的 `vlm.dets`）：`yolo.image`、`det_filter.dets`、
`seq_filter.dets`、`target_crop.frame/dets`、`annotate.image/dets`、`plate_recog.frame/crops`、
`face_detect.frame/crops`；**`vlm` 仅 `dets`**（报送哪些检测；图片源用 `image_source`，
`inputs.image` 不支持）。校验：未知产物 / 非法端口 / 非上游引用 /
类型不符 / 值非字符串 → 报错；同一产物被多个下游引用 → 允许（分支留路）。

## 一、主干生产者（向数据流放入新记录）

> 什么时候读：链路最上游——帧怎么进来、目标怎么被检出。

| 文档 | 模块 | 代码 | 主要 preset |
|---|---|---|---|
| [帧管理](frame_manager.md) | FrameManager 取帧节奏 + 背压（帧记录与原图旁路的起点） | `FrameManager` | FM0001~0003 |
| [YOLO 识别](yolo.md) | 小模型识别（**纯推理**，产 dict 检测记录，`kind=det`） | `YOLODetector` | YOLO0001~0003 |

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

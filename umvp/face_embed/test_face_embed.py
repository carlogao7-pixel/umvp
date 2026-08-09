# -*- coding: utf-8 -*-
"""
人脸向量化 + 底库检索命令行验证工具

演示管道: FaceDetector(检测+对齐) -> FaceEmbedder(512d) -> FaceStore(注册+检索)

用法示例:
    # 1) 批量注册：目录下每张图 = 一个身份，文件名(去扩展名) = 身份名
    python test_face_embed.py --register-dir id_photos/ --db out/face_db.npz

    # 2) 单图多人：检测出的每张脸注册为独立身份 {stem}_p{i}（适合合影切脸）
    python test_face_embed.py --register-face-image test1.jpg --db out/face_db.npz

    # 3) 指定身份注册单张图
    python test_face_embed.py --register-single Tom_Hanks.png --name tom_hanks --db out/face_db.npz

    # 4) 查询：图 -> 检测 -> 向量化 -> 检索
    python test_face_embed.py --query unknown.jpg --db out/face_db.npz --topk 5 --thresh 0.45

    # 5) 自检：底库每条向量重新检索，验证 top1 命中自身
    python test_face_embed.py --self-check --db out/face_db.npz

    # 6) 列出底库内容
    python test_face_embed.py --list --db out/face_db.npz
"""

import argparse
import sys
from pathlib import Path

import cv2

# 使 tinymodel_test 根目录可导入（face_detect 包）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_detect.face_detector import FaceDetector  # noqa: E402
from face_embed.face_embedder import FaceEmbedder  # noqa: E402
from face_embed.face_store import FaceStore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="人脸向量化与检索验证工具")
    p.add_argument("--register-dir", help="批量注册：目录下每张图=一个身份，文件名=身份")
    p.add_argument("--register-face-image", nargs="+", help="单图多人：每张脸注册为 {stem}_p{i}")
    p.add_argument("--register-single", nargs=2, metavar=("IMG", "NAME"), help="指定身份注册单图")
    p.add_argument("--query", help="查询图片路径")
    p.add_argument("--self-check", action="store_true", help="自检：底库每条向量检索自身")
    p.add_argument("--list", action="store_true", help="列出底库内容")
    p.add_argument("--db", default="out/face_db.npz", help="底库持久化文件（.npz）")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--thresh", type=float, default=0.45, help="检索相似度阈值")
    p.add_argument("--topk", type=int, default=5, help="返回 Top-K 结果")
    p.add_argument("--det-size", default="640,640", help="检测输入分辨率 宽,高")
    return p.parse_args()


def _det_size(s: str):
    w, h = s.split(",")
    return int(w.strip()), int(h.strip())


def _load_pipeline(device: str, det_size):
    det = FaceDetector(device=device, det_size=det_size)
    embedder = FaceEmbedder(device=device)
    return det, embedder


def _faces_of(det, embedder, img_path):
    """检测+向量化，返回 [(name, emb, bbox)]。"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [跳过] 无法读取: {img_path}")
        return []
    out = []
    for f in det.detect(img):
        if f.face_crop is None:
            continue
        emb = embedder.embed_face(f.face_crop)
        out.append((None, emb, f.bbox))
    return out


def do_register_dir(store, det, embedder, d: str) -> int:
    n = 0
    for p in sorted(Path(d).glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        faces = _faces_of(det, embedder, p)
        if not faces:
            print(f"  [跳过] 未检测到人脸: {p.name}")
            continue
        name = p.stem
        # 单人照：注册最高分那张脸；多人合影取每张脸注册同身份多向量
        for _, emb, _ in faces:
            store.register(name, emb, meta={"src": p.name})
            n += 1
        print(f"  [注册] {name}: {len(faces)} 张脸")
    return n


def do_register_face_image(store, det, embedder, imgs) -> int:
    n = 0
    for p in imgs:
        p = Path(p)
        faces = _faces_of(det, embedder, p)
        if not faces:
            print(f"  [跳过] 未检测到人脸: {p.name}")
            continue
        for i, (_, emb, bbox) in enumerate(faces):
            name = f"{p.stem}_p{i}"
            store.register(name, emb, meta={"src": p.name, "bbox": list(bbox)})
            n += 1
        print(f"  [注册] {p.name}: {len(faces)} 张脸")
    return n


def do_query(store, det, embedder, img_path: str, topk: int, thresh: float) -> None:
    faces = _faces_of(det, embedder, img_path)
    if not faces:
        print(f"[查询] 未检测到人脸: {img_path}")
        return
    print(f"[查询] {img_path} | 检测到 {len(faces)} 张人脸")
    for i, (_, emb, bbox) in enumerate(faces):
        print(f"  --- 人脸#{i} bbox={bbox} ---")
        results = store.search(emb, topk=topk, thresh=thresh)
        if not results:
            print("    无匹配（低于阈值）")
            continue
        for m in results:
            print(f"    #{m.rank} {m.name}  score={m.score:.3f}  {m.meta or ''}".rstrip())


def do_self_check(store) -> None:
    hit, total = 0, len(store)
    worst = (1.0, None)
    for i in range(total):
        r = store.search(store._embs[i], topk=3, thresh=0.0)
        top = r[0] if r else None
        if top and top.name == store._names[i]:
            hit += 1
        if top and top.score < worst[0]:
            worst = (top.score, store._names[i])
    print(f"[自检] {hit}/{total} 条 top1 命中自身")
    if total:
        print(f"  [自检] 最差命中分: {worst[0]:.3f} ({worst[1]})")


def main():
    args = parse_args()
    db_path = Path(args.db)
    store = FaceStore(thresh=args.thresh)
    if db_path.exists():
        store.load(db_path)
        print(f"[底库] 已加载: {db_path} | {store.stats()}")

    det, embedder = _load_pipeline(args.device, _det_size(args.det_size))
    registered = 0

    if args.register_dir:
        registered += do_register_dir(store, det, embedder, args.register_dir)
    if args.register_face_image:
        registered += do_register_face_image(store, det, embedder, args.register_face_image)
    if args.register_single:
        img, name = args.register_single
        faces = _faces_of(det, embedder, img)
        if faces:
            store.register(name, faces[0][1], meta={"src": Path(img).name})
            registered += 1
            print(f"  [注册] {name} <- {img}")
        else:
            print(f"  [跳过] 未检测到人脸: {img}")

    if registered:
        store.save(db_path)
        print(f"[底库] 已保存: {db_path} | {store.stats()}")

    if args.self_check:
        do_self_check(store)
    if args.list:
        print(f"[底库内容] {store.stats()}")
        for i, (name, meta) in enumerate(zip(store._names, store._metas)):
            print(f"  #{i} {name}  {meta or ''}".rstrip())
    if args.query:
        do_query(store, det, embedder, args.query, args.topk, args.thresh)

    if not any([args.register_dir, args.register_face_image, args.register_single,
                args.self_check, args.list, args.query]):
        print("请提供 --register-dir / --register-face-image / --register-single / "
              "--query / --self-check / --list 之一")


if __name__ == "__main__":
    main()

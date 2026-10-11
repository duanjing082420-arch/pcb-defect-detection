"""用训练好的 best.pt 对新图片做缺陷检测推理.

用法:
    python predict.py                                    # 默认检测 demo/原图/ 下的图片
    python predict.py --source path/to/img.jpg           # 对单张图预测
    python predict.py --source path/to/dir               # 对一个目录批量预测
    python predict.py --source path/to/dir --name my_run # 自定义输出子目录名
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

# 项目根目录 (相对脚本位置, 整个文件夹移到哪都能跑)
BASE = Path(__file__).resolve().parent.parent
# 训练好的最佳权重 (model/ 下只放这一个, 不会选错)
WEIGHTS = str(BASE / "model" / "best.pt")
# 检测结果输出根目录
OUT_DIR = str(BASE / "demo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(BASE / "demo" / "原图"),
                    help="图片/目录路径, 默认是 demo/原图/")
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    ap.add_argument("--name", default="检测结果", help="输出子目录名 (在 demo/ 下)")
    ap.add_argument("--save-txt", action="store_true", help="额外保存预测框坐标到 txt")
    ap.add_argument("--save", action="store_true", default=True, help="保存预测结果图")
    args = ap.parse_args()

    model = YOLO(WEIGHTS)
    print(f"✅ 已加载权重: {WEIGHTS}")
    print(f"📂 检测源: {args.source}")
    print(f"🔧 置信度阈值: {args.conf}\n")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        save_txt=args.save_txt,
        project=OUT_DIR,
        name=args.name,
        exist_ok=True,
    )

    # --- 汇总统计 ---
    total_objs = sum(len(r.boxes) for r in results)
    print(f"\n{'=' * 56}")
    print(f"共检测 {len(results)} 张图, 发现 {total_objs} 个缺陷目标")
    print(f"{'=' * 56}")

    # 只列出有缺陷的图, 便于快速筛查
    hit = [(Path(r.path).name, len(r.boxes)) for r in results if len(r.boxes) > 0]
    clean = len(results) - len(hit)
    print(f"\n有缺陷: {len(hit)} 张 | 无缺陷: {clean} 张\n")
    for name, n in hit:
        print(f"  ⚠️  {name}: {n} 个缺陷")

    print(f"\n结果图保存在: {OUT_DIR}\\{args.name}\\")


if __name__ == "__main__":
    main()
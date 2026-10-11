"""PCB缺陷检测 - YOLOv8 训练脚本 (GPU版).

环境: dxj (Python 3.9, torch 2.8.0+cu126, ultralytics 8.4.x)
硬件: NVIDIA GeForce RTX 4060 Laptop (8GB)
数据: 394张PCB图, 1类 [missing hole], 切分 train=315 / val=79
"""
import multiprocessing
from pathlib import Path
if __name__ == '__main__':
    multiprocessing.freeze_support()
    from ultralytics import YOLO
    import torch

    # --- 基础信息 ---
    print("=" * 60)
    print("PCB缺陷检测 - YOLOv8 训练")
    print("=" * 60)
    print(f"torch: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 配置 (相对脚本位置, 整个文件夹移到哪都能跑) ---
    BASE = Path(__file__).resolve().parent.parent           # 项目根目录
    DATA_YAML = str(BASE / "dataset" / "data.yaml")
    MODEL_NAME = str(BASE / "train_out" / "yolov8n.pt")     # 预训练权重
    PROJECT_DIR = str(BASE / "train_out")                   # 训练输出目录
    RUN_NAME = "pcb_train_v2"

    # --- 加载模型 (第一次跑会自动下载 yolov8n.pt 到当前目录) ---
    model = YOLO(MODEL_NAME)

    # --- 训练 ---
    results = model.train(
        data=DATA_YAML,
        epochs=100,            # 训练轮数 (8GB显存+小数据集, 100足够)
        imgsz=640,             # 输入尺寸
        batch=16,              # batch (8GB够用, 16更快)
        device=0,              # GPU 0
        workers=2,             # Windows下workers多了反而慢, 2~4合适
        cache=False,            # 315张小图, 直接读盘就行, 不必缓存
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,         # 同名run覆盖
        patience=20,           # 早停: 20轮没涨就停
        save_period=-1,        # -1=不存中间检查点(每次省216MB); 想断点续训就改回 10
        # --- 数据增强(单类小数据集适当加强) ---
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        flipud=0.0,            # PCB不建议上下翻转
        fliplr=0.5,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        # --- 输出 ---
        verbose=True,
        plots=True,            # 生成 results.png / confusion_matrix.png 等
    )

    print("\n训练完成! 结果保存在:")
    print(f"  {PROJECT_DIR}/{RUN_NAME}/")
    print(f"最佳权重: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
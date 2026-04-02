import pandas as pd
import matplotlib.pyplot as plt

csv_path = "data_n5q17_pathcheck/train_run/debug/debug_seed1_epoch1000/metrics.csv"
df = pd.read_csv(csv_path)

# x축: epoch 우선, 없으면 index
x_col = df["epoch"] if "epoch" in df.columns else df.index

# 원하는 4개 컬럼
target_cols = ["train_loss", "train_acc1", "valid_xe_loss", "valid_acc1"]

# 컬럼 존재 여부 체크
missing = [c for c in target_cols if c not in df.columns]
if missing:
    print(f"사용 가능한 컬럼: {list(df.columns)}")
    target_cols = [c for c in target_cols if c in df.columns]
    print(f"사용 가능한 컬럼만 시각화: {target_cols}")

# 그림 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.ravel()

for i, col in enumerate(target_cols):
    if i < 4:
        ax = axes[i]
        ax.plot(x_col, df[col], linewidth=1.8)
        ax.set_title(col)
        ax.grid(alpha=0.3)
        ax.set_ylabel(col)

axes[-1].set_xlabel("epoch" if "epoch" in df.columns else "index")
axes[-2].set_xlabel("epoch" if "epoch" in df.columns else "index")

plt.tight_layout()
out_path = "data_n5q17_pathcheck/train_run/debug/debug_seed1_epoch1000/metrics_4panel.png"
plt.savefig(out_path, dpi=150)
print(f"saved: {out_path}")

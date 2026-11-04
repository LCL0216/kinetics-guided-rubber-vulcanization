
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import argparse
import json
import hashlib


def resolve_dataset():
    parser = argparse.ArgumentParser(description="PG-ML Stage 4: 编号1–88")
    parser.add_argument("--project-root", default=None,
                        help="硫化曲线文件夹；可包含 Stage3_Output/ 子目录")
    parser.add_argument("--input", default=None,
                        help="明确指定 pgmlp_curve_dataset.csv 文件的完整路径")
    parser.add_argument("--repeats", type=int, default=5,
                        help="仅Stage4B/4C有效，重复5折的次数；正式默认5")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    candidates = []
    if args.input:
        fp = Path(args.input).expanduser().resolve()
        if not fp.is_file():
            raise FileNotFoundError(f"指定输入文件不存在：{fp}")
        return fp, fp.parent.parent if fp.parent.name == "Stage3_Output" else fp.parent, args
    bases = ([Path(args.project_root).expanduser().resolve()] if args.project_root else
             [script_dir, script_dir.parent, Path.cwd(), Path.home()/"Desktop"/"硫化曲线"])
    for base in bases:
        for fp in (base/"Stage3_Output"/"pgmlp_curve_dataset.csv",
                   base/"pgmlp_curve_dataset.csv",
                   base/"硫化曲线"/"Stage3_Output"/"pgmlp_curve_dataset.csv",
                   base/"建模集_1-88"/"Stage3_Output"/"pgmlp_curve_dataset.csv"):
            candidates.append(fp)
            if fp.is_file():
                root = fp.parent.parent if fp.parent.name == "Stage3_Output" else fp.parent
                return fp, root, args
    raise FileNotFoundError("找不到 Stage 3 数据集 pgmlp_curve_dataset.csv；已检查：\n" +
                            "\n".join(str(p) for p in candidates) +
                            "\n可以通过 --project-root 或 --input 指定路径")


def validate_model_dataset(df):
    required = ["No", "mixing_time", "mixing_temp", "curing_temp", "curing_pressure",
                "rotor_speed_1", "rotor_speed_2", "curing_time_offset",
                "ML", "MH", "ti", "k1", "n1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stage3数据缺少字段：{missing}")
    ids = pd.to_numeric(df["No"], errors="coerce")
    if ids.isna().any() or (ids != ids.round()).any() or len(ids) != 88 or sorted(ids.astype(int).tolist()) != list(range(1, 89)):
        raise ValueError("本阶段必须使用建模集88条曲线，No应为1–88各出现一次；请检查输入文件。")
    if df[required].isna().any().any():
        raise ValueError("Stage3核心字段存在缺失值，请先检查Stage3输出")
    print("已核对：建模集 No=1–88，88条样本，7个工艺输入与5个预测目标完整。")


def save_run_manifest(out_dir, input_file, script_name, settings):
    content = input_file.read_bytes()
    manifest = {"script": script_name, "input": str(input_file),
                "input_sha256": hashlib.sha256(content).hexdigest(),
                "output": str(out_dir), "dataset": "建模集_No1-88",
                "validation_set_included": False, "settings": settings}
    (out_dir/"stage4_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import torch
import torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

import joblib



RANDOM_STATE = 42

np.random.seed(
    RANDOM_STATE
)

torch.manual_seed(
    RANDOM_STATE
)



file_path, PROJECT_ROOT, ARGS = resolve_dataset()
out_dir = PROJECT_ROOT / "Stage4_Output"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"读取：{file_path}\n输出：{out_dir}")



df = pd.read_csv(
    file_path
)
validate_model_dataset(df)
save_run_manifest(out_dir, file_path, "Stage4.py", {"train_test_split": "80:20", "seed": 42, "architecture": "7-32-16-5", "max_epochs": 3000, "patience": 200})


print(
    "=" * 75
)

print(
    "PG-MLP Stage 4：第一版模型训练"
)

print(
    "=" * 75
)


print(
    f"样本数量：{len(df)}"
)

print(
    f"字段数量：{len(df.columns)}"
)



X_cols = [

    "mixing_time",

    "mixing_temp",

    "curing_temp",

    "curing_pressure",

    "rotor_speed_1",

    "rotor_speed_2",

    "curing_time_offset"
]


Y_cols = [

    "ML",

    "MH",

    "ti",

    "k1",

    "n1"
]


print(
    "\n输入 X："
)

print(
    X_cols
)


print(
    "\n输出 Y："
)

print(
    Y_cols
)



required_cols = (
    ["No"]
    +
    X_cols
    +
    Y_cols
)


missing_columns = [

    col

    for col in required_cols

    if col not in df.columns
]


if len(
    missing_columns
) > 0:

    raise ValueError(

        f"数据缺少字段："
        f"{missing_columns}"
    )


model_df = (
    df[
        required_cols
    ]
    .dropna()
    .copy()
)


print(
    f"\n去除缺失值后样本："
    f"{len(model_df)}"
)


# ============================================================
# 6. X / Y
# ============================================================

X = model_df[
    X_cols
].to_numpy(
    dtype=np.float32
)


Y = model_df[
    Y_cols
].to_numpy(
    dtype=np.float32
)


No = model_df[
    "No"
].to_numpy()



(
    X_train,
    X_test,
    Y_train,
    Y_test,
    No_train,
    No_test
) = train_test_split(

    X,

    Y,

    No,

    test_size=0.20,

    random_state=RANDOM_STATE
)


print(
    f"\n训练集数量："
    f"{len(X_train)}"
)

print(
    f"测试集数量："
    f"{len(X_test)}"
)


print(
    "\n测试集 No："
)

print(
    No_test
)



scaler_X = StandardScaler()

scaler_Y = StandardScaler()


X_train_scaled = scaler_X.fit_transform(
    X_train
)

X_test_scaled = scaler_X.transform(
    X_test
)


Y_train_scaled = scaler_Y.fit_transform(
    Y_train
)

Y_test_scaled = scaler_Y.transform(
    Y_test
)



joblib.dump(

    scaler_X,

    out_dir /
    "scaler_X.pkl"
)


joblib.dump(

    scaler_Y,

    out_dir /
    "scaler_Y.pkl"
)



X_train_tensor = torch.tensor(

    X_train_scaled,

    dtype=torch.float32
)


Y_train_tensor = torch.tensor(

    Y_train_scaled,

    dtype=torch.float32
)


X_test_tensor = torch.tensor(

    X_test_scaled,

    dtype=torch.float32
)


Y_test_tensor = torch.tensor(

    Y_test_scaled,

    dtype=torch.float32
)


# ============================================================
# 11. DataLoader
# ============================================================

train_dataset = TensorDataset(

    X_train_tensor,

    Y_train_tensor
)


train_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=True
)



class PGMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                output_dim
            )
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )



model = PGMLP(

    input_dim=len(
        X_cols
    ),

    output_dim=len(
        Y_cols
    )
)


print(
    "\n模型结构："
)

print(
    model
)



criterion = nn.MSELoss()



optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001,

    weight_decay=1e-4
)



MAX_EPOCHS = 3000

PATIENCE = 200

best_loss = np.inf

best_state = None

wait = 0


train_losses = []



print(
    "\n开始训练..."
)


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    epoch_losses = []


    for xb, yb in train_loader:

        optimizer.zero_grad()


        pred = model(
            xb
        )


        loss = criterion(
            pred,
            yb
        )


        loss.backward()


        optimizer.step()


        epoch_losses.append(
            loss.item()
        )


    mean_loss = float(
        np.mean(
            epoch_losses
        )
    )


    train_losses.append(
        mean_loss
    )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if mean_loss < best_loss - 1e-6:

        best_loss = mean_loss

        best_state = {

            k:
                v.clone()

            for k, v
            in model.state_dict().items()
        }

        wait = 0

    else:

        wait += 1


    if (
        epoch == 1
        or
        epoch % 100 == 0
    ):

        print(

            f"Epoch "
            f"{epoch:4d} | "

            f"Loss = "
            f"{mean_loss:.6f}"
        )


    if wait >= PATIENCE:

        print(

            f"\nEarly stopping "
            f"at epoch {epoch}"
        )

        break



if best_state is not None:

    model.load_state_dict(
        best_state
    )



torch.save(

    model.state_dict(),

    out_dir /
    "model.pth"
)



plt.figure(
    figsize=(6, 4)
)


plt.plot(
    train_losses
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "MSE Loss"
)


plt.title(
    "PG-MLP Training Loss"
)


plt.tight_layout()


plt.savefig(

    out_dir /
    "training_loss.png",

    dpi=300
)


plt.close()



def predict_original(
    model,
    X_scaled,
    scaler_y
):

    model.eval()


    with torch.no_grad():

        X_tensor = torch.tensor(

            X_scaled,

            dtype=torch.float32
        )


        pred_scaled = model(
            X_tensor
        ).numpy()


    pred = scaler_y.inverse_transform(
        pred_scaled
    )


    return pred



Y_train_pred = predict_original(

    model,

    X_train_scaled,

    scaler_Y
)



Y_test_pred = predict_original(

    model,

    X_test_scaled,

    scaler_Y
)



def evaluate_dataset(
    y_true,
    y_pred,
    dataset_name
):

    records = []


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        dataset_name
    )

    print(
        "=" * 70
    )


    for i, target in enumerate(
        Y_cols
    ):

        true = y_true[
            :, i
        ]

        pred = y_pred[
            :, i
        ]


        r2 = r2_score(
            true,
            pred
        )


        rmse = np.sqrt(
            mean_squared_error(
                true,
                pred
            )
        )


        mae = mean_absolute_error(
            true,
            pred
        )


        print(
            f"{target:8s} | "
            f"R² = {r2:8.4f} | "
            f"RMSE = {rmse:10.6f} | "
            f"MAE = {mae:10.6f}"
        )


        records.append({

            "Dataset":
                dataset_name,

            "Target":
                target,

            "R2":
                r2,

            "RMSE":
                rmse,

            "MAE":
                mae
        })


    return records



metrics = []


metrics += evaluate_dataset(

    Y_train,

    Y_train_pred,

    "Train"
)


metrics += evaluate_dataset(

    Y_test,

    Y_test_pred,

    "Test"
)


metrics_df = pd.DataFrame(
    metrics
)


metrics_df.to_csv(

    out_dir /
    "pgmlp_metrics.csv",

    index=False,

    encoding="utf-8-sig"
)



prediction_df = pd.DataFrame({

    "No":
        No_test
})


for i, target in enumerate(
    Y_cols
):

    prediction_df[
        f"{target}_true"
    ] = Y_test[
        :, i
    ]


    prediction_df[
        f"{target}_pred"
    ] = Y_test_pred[
        :, i
    ]


prediction_df.to_csv(

    out_dir /
    "pgmlp_predictions.csv",

    index=False,

    encoding="utf-8-sig"
)



for i, target in enumerate(
    Y_cols
):

    y_true = Y_test[
        :, i
    ]

    y_pred = Y_test_pred[
        :, i
    ]


    plt.figure(
        figsize=(5, 5)
    )


    plt.scatter(

        y_true,

        y_pred,

        s=55,

        alpha=0.75
    )


    min_value = min(

        np.min(
            y_true
        ),

        np.min(
            y_pred
        )
    )


    max_value = max(

        np.max(
            y_true
        ),

        np.max(
            y_pred
        )
    )


    plt.plot(

        [
            min_value,
            max_value
        ],

        [
            min_value,
            max_value
        ],

        "--",

        lw=1.5
    )


    r2 = r2_score(
        y_true,
        y_pred
    )


    rmse = np.sqrt(

        mean_squared_error(
            y_true,
            y_pred
        )
    )


    plt.text(

        0.05,

        0.92,

        f"R² = {r2:.3f}\n"
        f"RMSE = {rmse:.4g}",

        transform=
            plt.gca().transAxes,

        fontsize=10,

        va="top",

        bbox=dict(

            boxstyle=
                "round",

            alpha=
                0.7
        )
    )


    plt.xlabel(
        f"Experimental {target}"
    )


    plt.ylabel(
        f"Predicted {target}"
    )


    plt.title(
        f"{target}: Test Set"
    )


    plt.tight_layout()


    plt.savefig(

        out_dir /
        f"prediction_{target}.png",

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()



print(
    "\n"
    +
    "=" * 75
)

print(
    "Stage 4 完成！"
)

print(
    "=" * 75
)


print(
    f"\n训练样本："
    f"{len(X_train)}"
)


print(
    f"测试样本："
    f"{len(X_test)}"
)


print(
    "\n测试集结果："
)


test_metrics = metrics_df[

    metrics_df[
        "Dataset"
    ]
    ==
    "Test"
]


print(
    test_metrics
)


print(
    "\n输出位置："
    f"{out_dir}"
)


print(
    "=" * 75
)
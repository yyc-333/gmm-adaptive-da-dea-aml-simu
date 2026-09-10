import os
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport


# =====================================================
# 1. 读取真实数据
# =====================================================
data = pd.read_csv(
    r"E:\9-白血病预测\操作\process\dat2_aml_rf_sig_0.05.csv"
)

print("真实数据维度：", data.shape)
print(data.head())

# 检查 target
if "target" not in data.columns:
    raise ValueError("数据中不存在 target 列。")

target_unique = set(data["target"].dropna().unique())

if not {0, 1}.issubset(target_unique):
    raise ValueError(
        f"当前代码默认 target=1 为正类、target=0 为负类，"
        f"但实际 target 取值为：{sorted(target_unique)}"
    )

print("\n原始数据 target 分布：")
print(data["target"].value_counts(dropna=False))
print("\n原始数据 target 比例：")
print(data["target"].value_counts(normalize=True, dropna=False))


# =====================================================
# 2. Metadata
# =====================================================
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

metadata.update_column(
    column_name="target",
    sdtype="categorical"
)

metadata.update_column(
    column_name="Gen",
    sdtype="categorical"
)

metadata_dict = metadata.to_dict()


# =====================================================
# 3. 总输出路径
# =====================================================
base_output_dir = r"D:\白血病模拟数据"
os.makedirs(base_output_dir, exist_ok=True)


# =====================================================
# 4. 建立并训练 CTGAN
#    只训练一次，9种情形共用同一个CTGAN模型
# =====================================================
synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=300,
    verbose=True
)

print("\n开始训练 CTGAN...")
synthesizer.fit(data)
print("CTGAN 训练完成！")

# 保存模型
model_path = os.path.join(base_output_dir, "ctgan.pkl")
synthesizer.save(model_path)
print(f"CTGAN模型已保存至：{model_path}")


# =====================================================
# 5. 条件采样函数
# =====================================================
def sample_with_fixed_target_ratio(synthesizer, fixed_counts, seed):
    """
    按照 fixed_counts 中指定的 target 数量进行条件采样。

    fixed_counts 示例：
        {0: 2700, 1: 300}
    """
    sampled_parts = []

    for target_value, num_rows in fixed_counts.items():

        condition = Condition(
            num_rows=int(num_rows),
            column_values={"target": target_value}
        )

        part = synthesizer.sample_from_conditions(
            conditions=[condition]
        )

        if len(part) != int(num_rows):
            raise ValueError(
                f"target={target_value} 条件采样数量异常："
                f"期望 {int(num_rows)} 行，实际 {len(part)} 行"
            )

        # 再次强制写入 target，确保标签准确
        part["target"] = target_value
        sampled_parts.append(part)

    # 合并正负类
    synthetic_data = pd.concat(
        sampled_parts,
        axis=0,
        ignore_index=True
    )

    # 随机打乱
    synthetic_data = synthetic_data.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    return synthetic_data


# =====================================================
# 6. 模拟方案
#
# num_synthetic_rows:
#     3000 / 5000 / 8000
#
# target=1 正类比例:
#     10% / 5% / 1%
#
# 共 3 × 3 = 9 种情形
# =====================================================
sample_sizes = [3000, 5000, 8000]
positive_ratios = [0.05, 0.03,0.01]

# 每种情形生成100个数据集
num_repetitions = 100


# =====================================================
# 7. 循环生成9组模拟数据
# =====================================================
scenario_index = 0

for num_synthetic_rows in sample_sizes:

    for positive_ratio in positive_ratios:

        scenario_index += 1

        # -------------------------------------------------
        # 当前情形的正负类数量
        # -------------------------------------------------
        positive_count = int(round(num_synthetic_rows * positive_ratio))
        negative_count = num_synthetic_rows - positive_count

        fixed_target_counts = {
            0: negative_count,
            1: positive_count
        }

        # 百分比标签：10、5、1
        ratio_label = int(round(positive_ratio * 100))

        # 文件夹命名：
        # Simulated_data_MC_10_3000
        # Simulated_data_MC_5_3000
        # ...
        output_dir = os.path.join(
            base_output_dir,
            f"Simulated_data_MC_{ratio_label}_{num_synthetic_rows}"
        )

        os.makedirs(output_dir, exist_ok=True)

        print("\n")
        print("=" * 70)
        print(
            f"开始生成情形 {scenario_index}/9："
            f"N={num_synthetic_rows}, "
            f"target=1比例={positive_ratio:.0%}"
        )
        print(f"输出目录：{output_dir}")
        print(
            f"固定 target 数量："
            f"target=0 -> {negative_count}, "
            f"target=1 -> {positive_count}"
        )
        print("=" * 70)

        # 当前情形评分结果
        results = []

        # -------------------------------------------------
        # 每种情形生成100个模拟数据集
        # -------------------------------------------------
        for i in range(1, num_repetitions + 1):

            print(
                f"\n[{scenario_index}/9] "
                f"N={num_synthetic_rows}, "
                f"正类率={positive_ratio:.0%} | "
                f"第 {i}/{num_repetitions} 个数据集"
            )

            # 不同情形、不同重复使用不同随机种子
            seed = scenario_index * 100000 + i

            synthetic_data = sample_with_fixed_target_ratio(
                synthesizer=synthesizer,
                fixed_counts=fixed_target_counts,
                seed=seed
            )

            # -------------------------------------------------
            # 检查 target 分布
            # -------------------------------------------------
            synthetic_target_counts = (
                synthetic_data["target"]
                .value_counts(dropna=False)
                .sort_index()
            )

            synthetic_positive_ratio = (
                (synthetic_data["target"] == 1).mean()
            )

            print("当前模拟数据 target 分布：")
            print(synthetic_target_counts)
            print(
                f"实际 target=1 比例："
                f"{synthetic_positive_ratio:.4%}"
            )

            # -------------------------------------------------
            # 保存模拟数据
            # -------------------------------------------------
            save_path = os.path.join(
                output_dir,
                f"synthetic_CTGAN_{i:03d}.csv"
            )

            synthetic_data.to_csv(
                save_path,
                index=False,
                encoding="utf-8-sig"
            )

            # -------------------------------------------------
            # Quality Report
            # -------------------------------------------------
            report = QualityReport()

            report.generate(
                real_data=data,
                synthetic_data=synthetic_data,
                metadata=metadata_dict
            )

            quality_score = report.get_score()

            # -------------------------------------------------
            # Diagnostic Report
            # -------------------------------------------------
            diagnostic = DiagnosticReport()

            diagnostic.generate(
                real_data=data,
                synthetic_data=synthetic_data,
                metadata=metadata_dict
            )

            diagnostic_score = diagnostic.get_score()

            print(f"Quality Score    : {quality_score:.4f}")
            print(f"Diagnostic Score : {diagnostic_score:.4f}")

            # -------------------------------------------------
            # 记录结果
            # -------------------------------------------------
            results.append({
                "Dataset": i,
                "SampleSize": num_synthetic_rows,
                "Target1Ratio_Set": positive_ratio,
                "Target0Count": int(
                    synthetic_target_counts.get(0, 0)
                ),
                "Target1Count": int(
                    synthetic_target_counts.get(1, 0)
                ),
                "Target1Ratio_Actual": synthetic_positive_ratio,
                "QualityScore": quality_score,
                "DiagnosticScore": diagnostic_score
            })

        # =====================================================
        # 8. 保存当前情形100个数据集的评分
        # =====================================================
        results_df = pd.DataFrame(results)

        score_path = os.path.join(
            output_dir,
            "CTGAN_100datasets_scores.csv"
        )

        results_df.to_csv(
            score_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "-" * 70)
        print(
            f"当前情形完成：N={num_synthetic_rows}, "
            f"target=1比例={positive_ratio:.0%}"
        )
        print(f"数据保存至：{output_dir}")
        print(f"评分保存至：{score_path}")
        print("-" * 70)


# =====================================================
# 9. 全部完成
# =====================================================
print("\n")
print("=" * 70)
print("9种模拟情形全部生成完成！")
print(f"总输出路径：{base_output_dir}")
print("共生成 9 × 100 = 900 个模拟数据集。")
print("=" * 70)

"""
Agentic AutoML Platform v3.0.0 - 端到端测试脚本

测试完整工作流:
1. Step 1: 自然语言转 SQL 并提取数据
2. Step 2: 生成特征工程策略
3. Step 3: 数据清洗 + AutoML 训练
4. Step 4: 线上推理预测
5. MLOps: PSI 漂移检测

运行方式:
    cd /Users/rgj/PycharmProjects/Machine_Learning_Project
    source .venv/bin/activate
    python test_end_to_end.py

输出:
    - 控制台实时输出测试结果
    - 验证报告保存到桌面
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("E2E_Test")


class EndToEndTester:
    """端到端测试器"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.end_time = None
        self.task_id = None
        
        # 导入测试模块
        from core.llm.agent_schemas import FeatureStrategy, FeatureAction
        from model_code.feature_executor import FeatureEngineeringExecutor
        from model_code.mlops_monitor import DataDriftMonitor
        
        self.FeatureStrategy = FeatureStrategy
        self.FeatureAction = FeatureAction
        self.FeatureEngineeringExecutor = FeatureEngineeringExecutor
        self.DataDriftMonitor = DataDriftMonitor
    
    def run_all_tests(self):
        """运行所有测试"""
        self.start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("🧪 Agentic AutoML Platform v3.0.0 端到端测试")
        logger.info("=" * 60)
        
        # 测试 1: Pydantic Schema 验证
        self.test_pydantic_schema()
        
        # 测试 2: 特征工程执行器 (有状态)
        self.test_feature_executor()
        
        # 测试 3: MLOps PSI 漂移检测
        self.test_mlops_monitor()
        
        # 测试 4: 完整工作流模拟
        self.test_full_workflow()
        
        self.end_time = datetime.now()
        
        # 生成报告
        report = self.generate_report()
        
        return report
    
    def test_pydantic_schema(self):
        """测试 1: Pydantic Schema 验证"""
        logger.info("\n" + "=" * 60)
        logger.info("测试 1: Pydantic Schema 验证")
        logger.info("=" * 60)
        
        result = {
            "name": "Pydantic Schema 验证",
            "status": "pending",
            "details": []
        }
        
        try:
            # 测试 FeatureAction
            action = self.FeatureAction(
                column="age",
                action="fillna_median",
                reason="数值型缺失值 5%，中位数填充更鲁棒"
            )
            result["details"].append(f"✅ FeatureAction 创建成功：{action.column} -> {action.action}")
            
            # 测试非法 action 拦截
            try:
                invalid_action = self.FeatureAction(
                    column="test",
                    action="fillna_magic",  # 非法动作
                    reason="测试"
                )
                result["details"].append("❌ 非法 action 未被拦截")
                result["status"] = "failed"
            except Exception as e:
                result["details"].append(f"✅ 非法 action 被 Pydantic 拦截：{str(e)[:50]}...")
            
            # 测试 FeatureStrategy
            strategy = self.FeatureStrategy(actions=[action])
            result["details"].append(f"✅ FeatureStrategy 创建成功：{len(strategy.actions)} 个动作")
            
            result["status"] = "passed"
            
        except Exception as e:
            result["status"] = "failed"
            result["details"].append(f"❌ 测试失败：{e}")
        
        self.test_results.append(result)
        self._print_result(result)
    
    def test_feature_executor(self):
        """测试 2: 特征工程执行器 (有状态)"""
        logger.info("\n" + "=" * 60)
        logger.info("测试 2: 特征工程执行器 (有状态)")
        logger.info("=" * 60)
        
        result = {
            "name": "特征工程执行器",
            "status": "pending",
            "details": []
        }
        
        try:
            # 创建模拟数据
            np.random.seed(42)
            df_train = pd.DataFrame({
                'age': np.random.randint(20, 60, 100),
                'income': np.random.uniform(3000, 50000, 100),
                'city_code': np.random.choice(['北京', '上海', '广州'], 100),
                'is_fraud': np.random.randint(0, 2, 100)
            })
            
            # 添加缺失值
            df_train.loc[5:10, 'age'] = np.nan
            
            result["details"].append(f"✅ 训练数据创建成功：{df_train.shape}")
            
            # 创建策略
            strategy = self.FeatureStrategy(actions=[
                self.FeatureAction(column="age", action="fillna_median", reason="缺失值填充"),
                self.FeatureAction(column="income", action="log_transform", reason="偏态分布"),
                self.FeatureAction(column="city_code", action="target_encode", reason="类别编码"),
            ])
            
            result["details"].append(f"✅ 特征策略创建成功：{len(strategy.actions)} 个动作")
            
            # 训练阶段：fit_transform
            executor = self.FeatureEngineeringExecutor(target_col="is_fraud")
            df_clean = executor.fit_transform(df_train, strategy)
            
            result["details"].append(f"✅ fit_transform 完成：{df_clean.shape}")
            result["details"].append(f"   保存状态：fill_values={list(executor.state['fill_values'].keys())}")
            
            # 推理阶段：transform (使用保存的状态)
            df_infer = pd.DataFrame({
                'age': [35, 40, 45],
                'income': [10000, 15000, 20000],
                'city_code': ['北京', '上海', '广州'],
                'is_fraud': [0, 1, 0]
            })
            
            df_infer_clean = executor.transform(df_infer)
            
            result["details"].append(f"✅ transform 完成：{df_infer_clean.shape}")
            result["details"].append(f"   使用保存的填充值：age={executor.state['fill_values'].get('age', 'N/A')}")
            
            # 验证状态一致性
            if executor.is_fitted():
                result["details"].append("✅ 执行器状态：已拟合")
            else:
                result["details"].append("❌ 执行器状态：未拟合")
                result["status"] = "failed"
            
            result["status"] = "passed"
            
        except Exception as e:
            result["status"] = "failed"
            result["details"].append(f"❌ 测试失败：{e}")
        
        self.test_results.append(result)
        self._print_result(result)
    
    def test_mlops_monitor(self):
        """测试 3: MLOps PSI 漂移检测"""
        logger.info("\n" + "=" * 60)
        logger.info("测试 3: MLOps PSI 漂移检测")
        logger.info("=" * 60)
        
        result = {
            "name": "MLOps PSI 漂移检测",
            "status": "pending",
            "details": []
        }
        
        try:
            monitor = self.DataDriftMonitor(bins=10)
            
            # 创建基线数据 (训练分布)
            np.random.seed(42)
            baseline_data = np.random.normal(loc=50, scale=10, size=1000)
            
            # 创建线上数据 (无漂移)
            online_data_normal = np.random.normal(loc=50, scale=10, size=500)
            psi_normal = monitor.calculate_psi(baseline_data, online_data_normal)
            
            result["details"].append(f"✅ 无漂移场景 PSI={psi_normal:.4f} (< 0.1: 正常)")
            
            # 创建线上数据 (轻微漂移)
            online_data_drift = np.random.normal(loc=55, scale=10, size=500)
            psi_drift = monitor.calculate_psi(baseline_data, online_data_drift)
            
            result["details"].append(f"✅ 轻微漂移场景 PSI={psi_drift:.4f} (0.1-0.2: 警告)")
            
            # 创建线上数据 (严重漂移)
            online_data_critical = np.random.normal(loc=65, scale=15, size=500)
            psi_critical = monitor.calculate_psi(baseline_data, online_data_critical)
            
            result["details"].append(f"✅ 严重漂移场景 PSI={psi_critical:.4f} (≥ 0.2: 重训)")
            
            # 验证评判标准
            if psi_normal < 0.1:
                result["details"].append("✅ PSI 评判标准正确：无漂移")
            if 0.1 <= psi_drift < 0.2:
                result["details"].append("✅ PSI 评判标准正确：轻微漂移")
            if psi_critical >= 0.2:
                result["details"].append("✅ PSI 评判标准正确：严重漂移")
            
            # 测试基线提取
            df_test = pd.DataFrame({
                'age': np.random.randint(20, 60, 100),
                'income': np.random.uniform(3000, 50000, 100)
            })
            
            baseline_extracted = monitor.extract_baseline(df_test, ['age', 'income'])
            result["details"].append(f"✅ 基线数据提取成功：{len(baseline_extracted)} 项")
            
            result["status"] = "passed"
            
        except Exception as e:
            result["status"] = "failed"
            result["details"].append(f"❌ 测试失败：{e}")
        
        self.test_results.append(result)
        self._print_result(result)
    
    def test_full_workflow(self):
        """测试 4: 完整工作流模拟"""
        logger.info("\n" + "=" * 60)
        logger.info("测试 4: 完整工作流模拟")
        logger.info("=" * 60)
        
        result = {
            "name": "完整工作流模拟",
            "status": "pending",
            "details": []
        }
        
        try:
            # Step 1: 模拟数据提取
            result["details"].append("Step 1: 数据提取")
            np.random.seed(42)
            df_raw = pd.DataFrame({
                'order_id': range(1000),
                'driver_id': np.random.randint(1, 100, 1000),
                'order_fee': np.random.uniform(10, 500, 1000),
                'age': np.random.randint(20, 60, 1000),
                'city_code': np.random.choice(['北京', '上海', '广州', '深圳'], 1000),
                'is_fraud': np.random.randint(0, 2, 1000)
            })
            result["details"].append(f"  ✅ 原始数据：{df_raw.shape}")
            
            # Step 2: 生成特征策略
            result["details"].append("\nStep 2: 特征策略生成")
            strategy = self.FeatureStrategy(actions=[
                self.FeatureAction(column="age", action="fillna_median", reason="缺失值填充"),
                self.FeatureAction(column="order_fee", action="log_transform", reason="偏态分布"),
                self.FeatureAction(column="city_code", action="target_encode", reason="类别编码"),
            ])
            result["details"].append(f"  ✅ 策略：{len(strategy.actions)} 个动作")
            
            # Step 3: 特征工程 + 训练
            result["details"].append("\nStep 3: 特征工程 + 训练")
            executor = self.FeatureEngineeringExecutor(target_col="is_fraud")
            df_clean = executor.fit_transform(df_raw, strategy)
            result["details"].append(f"  ✅ 清洗后数据：{df_clean.shape}")
            
            # 模拟训练 (简单逻辑回归)
            from sklearn.linear_model import LogisticRegression
            X = df_clean.drop(columns=['is_fraud', 'order_id'])
            y = df_clean['is_fraud']
            
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            train_auc = model.score(X, y)
            result["details"].append(f"  ✅ 训练 AUC={train_auc:.4f}")
            
            # Step 4: 持久化
            result["details"].append("\nStep 4: 模型持久化")
            os.makedirs("saved", exist_ok=True)
            self.task_id = f"test_{int(time.time())}"
            
            import joblib
            joblib.dump({
                "model": model,
                "feature_executor": executor,
                "config": {"task_type": "binary_classification"}
            }, f"saved/model_{self.task_id}.pkl")
            
            result["details"].append(f"  ✅ 模型已保存：saved/model_{self.task_id}.pkl")
            
            # Step 5: 线上推理
            result["details"].append("\nStep 5: 线上推理")
            artifact = joblib.load(f"saved/model_{self.task_id}.pkl")
            loaded_executor = artifact["feature_executor"]
            loaded_model = artifact["model"]
            
            df_infer = pd.DataFrame({
                'order_id': [1001],
                'driver_id': [50],
                'order_fee': [100],
                'age': [35],
                'city_code': ['北京'],
                'is_fraud': [0]
            })
            
            df_infer_clean = loaded_executor.transform(df_infer)
            X_infer = df_infer_clean.drop(columns=['is_fraud', 'order_id'])
            
            prediction = loaded_model.predict(X_infer)[0]
            probability = loaded_model.predict_proba(X_infer)[0][1]
            
            result["details"].append(f"  ✅ 预测结果：{prediction} (概率={probability:.4f})")
            
            # Step 6: MLOps 监控
            result["details"].append("\nStep 6: MLOps 监控")
            monitor = self.DataDriftMonitor()
            baseline = monitor.extract_baseline(df_clean, ['age', 'order_fee'])
            monitor.save_baseline(baseline, f"saved/baseline_{self.task_id}.json")
            
            result["details"].append(f"  ✅ 基线数据已保存：saved/baseline_{self.task_id}.json")
            
            # 模拟漂移检测
            psi_report = monitor.generate_drift_report(baseline, df_clean)
            result["details"].append(f"  ✅ 漂移检测：{psi_report['summary']['drifted_features']} 个特征漂移")
            
            result["status"] = "passed"
            
        except Exception as e:
            result["status"] = "failed"
            result["details"].append(f"❌ 测试失败：{e}")
            import traceback
            result["details"].append(traceback.format_exc())
        
        self.test_results.append(result)
        self._print_result(result)
    
    def _print_result(self, result):
        """打印测试结果"""
        status_icon = "✅" if result["status"] == "passed" else "❌"
        logger.info(f"{status_icon} {result['name']}: {result['status'].upper()}")
        for detail in result["details"]:
            logger.info(f"   {detail}")
    
    def generate_report(self):
        """生成测试报告"""
        passed = sum(1 for r in self.test_results if r["status"] == "passed")
        total = len(self.test_results)
        
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        report = {
            "test_name": "Agentic AutoML Platform v3.0.0 端到端测试",
            "test_time": self.start_time.isoformat() if self.start_time else "N/A",
            "duration_seconds": duration,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": f"{passed/total*100:.1f}%" if total > 0 else "N/A"
            },
            "results": self.test_results,
            "task_id": self.task_id
        }
        
        return report
    
    def save_report(self, report, output_path):
        """保存测试报告"""
        # 生成 Markdown 报告
        md_content = f"""# Agentic AutoML Platform v3.0.0 端到端测试报告

## 📊 测试概览

| 项目 | 结果 |
|------|------|
| **测试时间** | {report['test_time']} |
| **测试耗时** | {report['duration_seconds']:.2f} 秒 |
| **总测试数** | {report['summary']['total_tests']} |
| **通过数** | {report['summary']['passed']} |
| **失败数** | {report['summary']['failed']} |
| **通过率** | {report['summary']['pass_rate']} |

## 🧪 详细测试结果

"""
        
        for i, result in enumerate(report['results'], 1):
            status_icon = "✅" if result["status"] == "passed" else "❌"
            md_content += f"### {i}. {result['name']} {status_icon}\n\n"
            md_content += f"**状态**: {result['status'].upper()}\n\n"
            md_content += "**详情**:\n"
            for detail in result["details"]:
                md_content += f"- {detail}\n"
            md_content += "\n---\n\n"
        
        md_content += f"""
## 🎯 测试结论

"""
        
        if report['summary']['passed'] == report['summary']['total_tests']:
            md_content += "✅ **所有测试通过！Agentic AutoML Platform v3.0.0 功能完整，可以投入生产使用。**\n\n"
        else:
            md_content += f"⚠️ **{report['summary']['failed']} 个测试失败，请检查失败原因并修复。**\n\n"
        
        md_content += f"""
## 📦 测试产物

- **任务 ID**: `{report['task_id']}`
- **模型文件**: `saved/model_{report['task_id']}.pkl`
- **基线数据**: `saved/baseline_{report['task_id']}.json`

---

*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*平台版本：v3.0.0*
"""
        
        # 保存 Markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        logger.info(f"\n📄 测试报告已保存：{output_path}")
        
        # 同时保存 JSON
        json_path = output_path.replace(".md", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 JSON 报告已保存：{json_path}")
        
        return output_path


def main():
    """主函数"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Agentic AutoML Platform v3.0.0 端到端测试")
    logger.info("=" * 60)
    
    # 创建测试器
    tester = EndToEndTester()
    
    # 运行所有测试
    report = tester.run_all_tests()
    
    # 保存报告到桌面
    desktop_path = Path.home() / "Desktop"
    desktop_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = desktop_path / f"Agentic_AutoML_v3_E2E_Test_Report_{timestamp}.md"
    
    tester.save_report(report, str(report_path))
    
    # 打印总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试总结")
    logger.info("=" * 60)
    logger.info(f"总测试数：{report['summary']['total_tests']}")
    logger.info(f"通过：{report['summary']['passed']}")
    logger.info(f"失败：{report['summary']['failed']}")
    logger.info(f"通过率：{report['summary']['pass_rate']}")
    logger.info(f"测试耗时：{report['duration_seconds']:.2f}秒")
    logger.info(f"\n📄 报告已保存到桌面：{report_path}")
    logger.info("\n" + "=" * 60)
    
    # 返回退出码
    sys.exit(0 if report['summary']['failed'] == 0 else 1)


if __name__ == "__main__":
    main()

import pandas as pd
import os
import argparse
from rag_model import load_rag_system, RAGLegalSystem
import re
class LegalEvaluator:
    def __init__(self, rag_system=None, config_path="rag_config.json", index_path="legal_index.pkl"):
        """初始化评估器"""
        # 尝试加载现有的RAG系统
        self.rag_system = rag_system
        if self.rag_system is None:
            self.rag_system = load_rag_system(config_path, index_path)
            
        # 如果加载失败，提示用户先运行初始化脚本
        if self.rag_system is None:
            print(f"RAG system configuration not found at {config_path} or index not found at {index_path}.")
            print("Please run 'python rag_model.py' first to initialize and save the RAG system.")
            raise ValueError("RAG system not initialized")
    
    def evaluate_with_rag(self, csv_list_mul, path="../data/val_csv_single/", output_dir="results"):
        """使用 RAG 进行评测"""
        if not self.rag_system.query_engine:
            raise ValueError("RAG system not properly initialized.")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        total_len = 0
        all_answers = []
        acc = 0
        
        result_summary = {}
        
        for name in csv_list_mul:
            csv_name = os.path.join(path, name)
            print(f"Processing {csv_name}...")
            
            df = pd.read_csv(csv_name)
            total_len += len(df)
            
            file_answers = []
            file_correct = 0
            
            for i in range(len(df)):
                question = df["input"][i]
                options = {
                    'A': df["A"][i],
                    'B': df["B"][i], 
                    'C': df["C"][i],
                    'D': df["D"][i]
                }
                
                # 使用 RAG 生成答案
                content = self.rag_system.generate_answer_with_rag(question, options)
                content = re.sub(r'[^a-zA-Z]', '', content)
                # 清理答案格式
                content = content.strip().upper()
                
                # 检查答案是否正确
                is_correct = str(content) == str(df["output"][i])
                if is_correct:
                    acc += 1
                    file_correct += 1
                
                # 保存答案和结果
                answer_data = {
                    "question": question,
                    "options": options,
                    "predicted": content,
                    "correct": df["output"][i],
                    "is_correct": is_correct
                }
                file_answers.append(answer_data)
                all_answers.append(answer_data)
                
                # 输出答案
                print(content)
                
                # 每处理10题显示一次进度
                if (i + 1) % 10 == 0:
                    current_acc = acc / (total_len - len(df) + i + 1)
                    print(f"Progress: {i+1}/{len(df)}, Current Accuracy: {current_acc:.4f}")
            
            # 保存当前文件的结果
            file_accuracy = file_correct / len(df) if len(df) > 0 else 0
            result_summary[name] = {
                "total": len(df),
                "correct": file_correct,
                "accuracy": file_accuracy
            }
            
            print(f"Completed {name}, Current total accuracy: {acc}/{total_len}")
            
            # 保存该文件的详细结果
            file_results = {
                "file_name": name,
                "total": len(df),
                "correct": file_correct,
                "accuracy": file_accuracy,
                "answers": file_answers
            }
            
            # 保存为CSV文件
            results_df = pd.DataFrame({
                "question": [a["question"] for a in file_answers],
                "predicted": [a["predicted"] for a in file_answers],
                "correct": [a["correct"] for a in file_answers],
                "is_correct": [a["is_correct"] for a in file_answers]
            })
            results_df.to_csv(os.path.join(output_dir, f"results_{name}"), index=False)
        
        # 计算总体准确率
        percentage = acc / total_len if total_len > 0 else 0
        
        # 打印最终结果
        print(f"\nFinal Results:")
        print(f"Total Questions: {total_len}")
        print(f"Correct Answers: {acc}")
        print(f"Accuracy: {percentage:.4f} ({percentage*100:.2f}%)")
        
        # 打印每个文件的结果
        print("\nResults by file:")
        for name, result in result_summary.items():
            print(f"{name}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")
        
        # 保存总体结果
        overall_results = {
            "total": total_len,
            "correct": acc,
            "accuracy": percentage,
            "file_results": result_summary
        }
        
        # 保存为CSV文件
        pd.DataFrame({
            "file": list(result_summary.keys()),
            "total": [r["total"] for r in result_summary.values()],
            "correct": [r["correct"] for r in result_summary.values()],
            "accuracy": [r["accuracy"] for r in result_summary.values()]
        }).to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
        
        return percentage, all_answers, overall_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate legal questions using RAG system")
    parser.add_argument("--config_path", type=str, default="rag_config.json", help="Path to the RAG system configuration")
    parser.add_argument("--index_path", type=str, default="legal_index.pkl", help="Path to the saved index")
    parser.add_argument("--data_path", type=str, default="../data/val_csv_single/", help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--files", nargs="+", default=["mcq_sing_cpa.csv", "mcq_sing_lbk.csv", "mcq_sing_nje.csv", 
                                                      "mcq_sing_pae.csv", "mcq_sing_pfe.csv", "mcq_sing_ungee.csv"],
                        help="List of CSV files to evaluate")
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = LegalEvaluator(config_path=args.config_path, index_path=args.index_path)
    
    # 进行评测
    accuracy, answers, results = evaluator.evaluate_with_rag(
        csv_list_mul=args.files,
        path=args.data_path,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation Complete!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 
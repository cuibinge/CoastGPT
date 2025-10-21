"""
该脚本用于对模型视觉问答的预测结果进行评估：
1. 加载模型预测结果 json 文件（包含预测答案(pred)/真实答案(target)/问题类型(types)）。
2. 使用 TextVQAAccuracyEvaluator 计算每个预测是否正确。
3. 按照问题类型统计 correct、total、accuracy。
4. 输出总体准确率，并保存为评估结果文件（json）。
"""
import os
import re
import json
import logging
from collections import defaultdict
from typing import List
from tqdm import tqdm

# 设置日志输出
logger = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO)

# ======================
# 答案预处理类：EvalAIAnswerProcessor
# ======================
class EvalAIAnswerProcessor:
    """
    EvalAIAnswerProcessor 类用于处理答案文本，使其更准确。
    功能包括：
    - 小写化、去除标点
    - 数字映射（例如 "two" -> "2"）
    - 去除冠词（a, an, the）
    - 统一缩写形式（例如 "cant" -> "can't"）
    """

    # 常见缩写映射
    CONTRACTIONS = {
        "aint": "ain't", "arent": "aren't", "cant": "can't",
        "couldve": "could've", "couldnt": "couldn't",
        "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
        "hes": "he's", "Im": "I'm", "Ive": "I've",
        "isnt": "isn't", "itd": "it'd", "itll": "it'll",
        "lets": "let's", "shant": "shan't", "shouldnt": "shouldn't",
        "thats": "that's", "theres": "there's", "theyre": "they're",
        "theyve": "they've", "wasnt": "wasn't", "weve": "we've",
        "wont": "won't", "wouldnt": "wouldn't", "yall": "y'all",
        "youd": "you'd", "youll": "you'll", "youre": "you're",
        "youve": "you've"
    }

    # 数字词映射
    NUMBER_MAP = {
        "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10"
    }

    # 冠词列表
    ARTICLES = ["a", "an", "the"]

    # 用于处理标点符号的正则
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+",
                    "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def __call__(self, item):
        """
        处理输入文本，返回清洗后的标准化答案
        """
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item

    def word_tokenize(self, word):
        """小写化，去除部分符号"""
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        """去除标点符号"""
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (re.search(self.COMMA_STRIP, in_text) is not None):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        """数字映射 + 去掉冠词 + 统一缩写"""
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)  # 数字词映射
            if word not in self.ARTICLES:  # 去掉冠词
                out_text.append(word)
        for word_id, word in enumerate(out_text):  # 缩写统一
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        return " ".join(out_text)

# ======================
# 准确率评估类
# ======================
class TextVQAAccuracyEvaluator:
    """
    TextVQAAccuracyEvaluator 类用于评估预测答案的准确率
    """

    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        计算答案的 soft score。
        raw_answers: 可能是一个列表（多个参考答案），也可能是单个答案。
        """
        unique_answer_scores = {}
        if isinstance(raw_answers, List):
            answers = [self.answer_processor(a) for a in raw_answers]
            gt_answers = list(enumerate(answers))
            unique_answers = set(answers)
        else:
            unique_answer_scores[raw_answers] = 1
            return unique_answer_scores

        # 遍历每个可能答案，计算其得分
        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == unique_answer]
                acc = min(1, float(len(matching_answers)) / 3)  # VQA 标准：至少有3个一致才算满分
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores


# ======================
# 主评估函数
# ======================
def evaluate_result_file(result_file: str, output_json: str = None):
    """
    输入：模型预测结果文件（json）
    输出：总准确率和分类型指标，并可保存为新的评估结果 json 文件
    """
    # 加载预测结果
    with open(result_file, "r") as f:
        predictions = json.load(f)

    evaluator = TextVQAAccuracyEvaluator()
    pred_scores = []
    diff_type_score = defaultdict(list)

    # 遍历每个预测，计算得分
    for entry in tqdm(predictions):
        pred_answer = evaluator.answer_processor(entry["pred"])  # 模型预测答案
        unique_answer_scores = evaluator._compute_answer_scores(entry["target"])  # GT 答案得分表
        score = unique_answer_scores.get(pred_answer, 0.0)
        if score == 0.0 and pred_answer in entry["target"]:  # 保底：预测在 GT 列表里算 1
            score = 1.0
        pred_scores.append(score)
        diff_type_score[entry["types"]].append(score)  # 按类型存储分数

    # 每类结果统计
    results_dict = {}
    for t, scores in diff_type_score.items():
        correct = int(sum(scores))
        total = len(scores)
        acc = correct / total if total > 0 else 0
        results_dict[t] = {
            "correct": correct,
            "total": total,
            "accuracy": acc
        }

    # 总体结果统计
    total_acc = sum(pred_scores) / len(pred_scores) if len(pred_scores) > 0 else 0
    results_dict["Total"] = {"accuracy": total_acc}

    # 输出日志
    logger.info(f"Total Accuracy: {100.0 * total_acc:.2f}%")
    print(f"Total Accuracy: {100.0 * total_acc:.2f}%")

    # 保存结果文件
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Saved results to {output_json}")
        print(f"Saved results to {output_json}")

# ======================
# 命令行接口
# ======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, required=True,
                        help="模型推理结果 json 文件，例如 output/RSVQA_HR_result.json")
    parser.add_argument("--output-json", type=str, default=None,
                        help="评估指标保存路径（json），例如 output/RSVQA_HR_metrics.json")
    args = parser.parse_args()

    evaluate_result_file(args.result_file, args.output_json)





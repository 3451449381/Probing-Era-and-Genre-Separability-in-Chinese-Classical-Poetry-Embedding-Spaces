import os
import logging
import csv
import numpy as np

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)

gpu_list = None


class Bert(nn.Module):
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.max_length = 512  # 设置最大长度

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, cls=False):
        result = []
        x = data['input_ids']
        y = self.bert(input_ids=x, 
                      attention_mask=data['attention_mask'],
                      token_type_ids=data['token_type_ids'])[0]
        
        if cls:
            result = y[:, 0, :].view(y.size(0), -1)
            result = result.cpu().tolist()
        else:
            result = []
            y = y.cpu()
            for i in range(y.shape[0]):
                # 获取有效token的均值
                mask = data['attention_mask'][i].cpu()
                if torch.sum(mask) > 2:  # 确保有有效token
                    tmp = y[i][1:torch.sum(mask) - 1, :]
                    result.append(tmp.mean(0).tolist())
                else:
                    result.append(y[i][0, :].tolist())  # 使用CLS token

        return result


class BertFormatter():
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.max_length = 512  # BERT的最大长度

    def process(self, data):
        # 使用新的padding参数
        res_dict = self.tokenizer.batch_encode_plus(
            data, 
            padding=True,  # 替代pad_to_max_length
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'  # 直接返回PyTorch张量
        )

        input_list = {
            'input_ids': res_dict['input_ids'],
            'attention_mask': res_dict['attention_mask'],
            'token_type_ids': res_dict.get('token_type_ids', torch.zeros_like(res_dict['input_ids']))
        }
        return input_list


def init(BERT_PATH="./BERT_CCPoem_v1"):
    global gpu_list
    gpu_list = []

    device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if device_list[0] == "":
        device_list = []
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

    cuda = torch.cuda.is_available()
    logging.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logging.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    model = Bert(BERT_PATH)
    formatter = BertFormatter(BERT_PATH)
    if len(gpu_list) > 0:
        model = model.cuda()
    if len(gpu_list) > 1:
        try:
            model.init_multi_gpu(gpu_list)
        except Exception as e:
            logging.warning(
                "No init_multi_gpu implemented in the model, use single gpu instead. {}".format(str(e)))
    return model, formatter


def predict_vec_rep(data, model, formatter):
    data = formatter.process(data)
    model.eval()

    # 移动到GPU（如果可用）
    if len(gpu_list) > 0:
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

    with torch.no_grad():  # 增加无梯度计算，节省内存
        result = model(data)

    return result


def cos_sim(vector_a, vector_b, sim=True):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        return 0.0 if sim else -1.0
    cos = num / denom
    if not sim:
        return cos
    sim = 0.5 + 0.5 * cos
    return sim


def read_poetry_data(file_path="诗歌数据.csv"):
    """读取诗歌数据CSV文件，返回内容列表"""
    poetry_contents = []
    try:
        # 先尝试用竖线分隔，如果不行再用逗号
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取前几行检测分隔符
            sample_lines = []
            for i in range(5):
                try:
                    line = next(f)
                    sample_lines.append(line)
                except StopIteration:
                    break
            f.seek(0)  # 回到文件开头
            
            # 检测实际使用的分隔符
            delimiter = '|' if any('|' in line for line in sample_lines) else ','
            logging.info(f"使用分隔符: {repr(delimiter)}")
            
            reader = csv.reader(f, delimiter=delimiter)
            next(reader)  # 跳过表头
            
            for idx, row in enumerate(reader):
                if len(row) >= 3:  # 至少包含内容列
                    try:
                        # 根据实际列数处理
                        if len(row) >= 5:
                            title, author, content, dynasty, genre = row[:5]
                        else:
                            title, author, content = row[:3]
                        
                        content = content.strip()
                        if content:  # 确保内容不为空
                            poetry_contents.append(content)
                    except Exception as e:
                        logging.warning(f"处理第{idx+2}行出错: {str(e)}")
                else:
                    logging.warning(f"第{idx+2}行列数不足: {row}")
            
            logging.info(f"成功读取 {len(poetry_contents)} 首诗歌")
            return poetry_contents
            
    except FileNotFoundError:
        logging.error(f"未找到文件: {file_path}")
        raise
    except Exception as e:
        logging.error(f"读取文件出错: {str(e)}")
        raise


def generate_poetry_vectors(batch_size=16):  # 减小批次大小
    """生成诗歌向量并保存为numpy文件"""
    # 读取诗歌数据
    poetry_contents = read_poetry_data()
    if not poetry_contents:
        logging.error("没有有效的诗歌数据，无法生成向量")
        return

    # 初始化模型
    model, formatter = init()
    
    # 批量处理生成向量
    all_vectors = []
    total = len(poetry_contents)
    
    for i in range(0, total, batch_size):
        batch = poetry_contents[i:i+batch_size]
        logging.info(f"处理第 {i+1}-{min(i+batch_size, total)} 首诗歌")
        vectors = predict_vec_rep(batch, model, formatter)
        all_vectors.extend(vectors)
    
    # 转换为numpy数组并保存
    if all_vectors:
        vectors_np = np.array(all_vectors)
        np.save("output.npy", vectors_np)
        logging.info(f"成功生成 {vectors_np.shape[0]} × {vectors_np.shape[1]} 的向量文件，保存为 output.npy")
    else:
        logging.error("未能生成任何向量")


if __name__ == '__main__':
    generate_poetry_vectors()

if __name__ == '__main__1':
    model, formatter = init()
    result_0 = predict_vec_rep(["一行白鹭上青天"], model, formatter)[0]
    result_1 = predict_vec_rep(['白鹭一行登碧霄'], model, formatter)[0]
    result_2 = predict_vec_rep(["飞却青天白鹭鸶"], model, formatter)[0]

    print(cos_sim(result_0, result_1))
    print(cos_sim(result_0, result_2))

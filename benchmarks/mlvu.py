import json
import os
import pandas as pd

class MLVU:
    def __init__(self, config):
        self.video_dir = config['video_dir']
        self.raw_anno_path = config['anno_path']
        self.subset = config['subset']

        assert self.subset == "dev"

        df = pd.read_parquet(self.raw_anno_path)
        data = df.to_json(orient="records")
        data = json.loads(data)
        self.rows = []
        row_id = 0
        for i in range(len(data)):
            video_path = os.path.join(self.video_dir, data[i]['video_name'])
            question = "Question: " + data[i]['question'].split('\n')[0] + "\n"
            question += "Options:\n"
            for idx, c in enumerate(data[i]['candidates']):
                question += f"{chr(ord('A') + idx)}. {c}\n"
            question = question.rstrip()
            full_prompt = "Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
            full_prompt += "\n" + question + "\n"
            full_prompt += "Only give the best option.\nBest Option: "
            self.rows.append({
                "id" : row_id,
                "video_path" : video_path,
                "ques" : data[i]['question'].split('\n')[0],
                "opts" : data[i]['candidates'],
                "gt_ans" : data[i]['answer'],
                "task_type" : data[i]['task_type'],
                # "full_prompt" : data[i]['question'] + "\nOnly give the best option.\nBest option: (",
                "full_prompt" : full_prompt
            })
            row_id+=1
    def metrics(self, result_path):
        type_map = {}
        for i in range(len(self.rows)):
            type_map[self.rows[i]['id']] = self.rows[i]['task_type']
        res = {}
        total,count = 0,0
        with open(result_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                task_type = type_map[item['QA']['id']]
                if task_type not in res:
                    res[task_type]=[0,0,0]
                if item['ans'][0]==item['QA']['gt_ans'][0]:
                    res[task_type][0]+=1
                    count+=1
                total+=1
                res[task_type][1]+=1
                res[task_type][2]=res[task_type][0]/res[task_type][1]
        acc = 0
        for v in res.values():
            acc += v[2]
        acc /= len(res)
        res['M-Avg'] = acc
        res['overall'] = {
            "true_count" : count,
            "total_count" : total,
            "accuracy" : count/total,
        }
        return res
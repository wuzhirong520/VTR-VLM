import json
import os
import pandas as pd

class VideoEvalPro:
    def __init__(self, config):
        self.video_dir = config['video_dir']
        self.raw_anno_path = config['anno_path']
        self.subset = config['subset']

        df = pd.read_parquet(self.raw_anno_path)
        data = df.to_json(orient="records")
        data = json.loads(data)

        self.rows = []
        self.question_types_map = {}
        self.question_types = set()
        row_id = 0
        for i in range(len(data)):
            video_path = os.path.join(self.video_dir, data[i]['video'])
            oe_question = data[i]['question'] + " Keep the answer short and concise."
            mcq_question = '\n'.join([
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option, with no text around it.',
                data[i]['question'], ' '.join(data[i]['options'])
            ])
            self.question_types.add(data[i]['qa_type'])
            self.question_types_map[row_id] = data[i]['qa_type']
            self.rows.append({
                "id" : row_id,
                "video_path" : video_path,
                "ques" : data[i]['question'],
                "opts" :[o[3:] for o in data[i]['options']],
                "gt_ans" : data[i]['answer'],
                "full_prompt" : oe_question if self.subset=="open" else mcq_question
            })
            row_id+=1
    def metrics(self, result_path):
        if self.subset=="open":
            runned_ids = []
            with open(result_path, "r") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    runned_ids.append(item['QA']['id'])
            return {"total_count" : len(runned_ids)}
        else:
            res = {qatype : {"true_count":0,"total_count":0} for qatype in self.question_types}
            res['overall'] = {"true_count":0,"total_count":0}
            with open(result_path, "r") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    is_true = False
                    if item['ans'][0]==item['QA']['gt_ans'][0]:
                        res['overall']['true_count']+=1
                        is_true = True
                    res['overall']['total_count']+=1
                    qatype = self.question_types_map[item['QA']['id']]
                    res[qatype]['total_count']+=1
                    if is_true:
                        res[qatype]['true_count']+=1
            for qatype in res:
                res[qatype]['accuracy'] = res[qatype]['true_count']/res[qatype]['total_count'] if res[qatype]['total_count']>0 else 0
            return res
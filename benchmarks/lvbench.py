import json
import os

class LVBench:
    def __init__(self, config):
        self.video_dir = config['video_dir']
        self.raw_anno_path = config['anno_path']

        with open(self.raw_anno_path, "r") as f:
            data = []
            for line in f.readlines():
                item = json.loads(line)
                data.append(item)


        # ############################
        # with open("/mnt/bn/wzr/code/VTR-VLM/logs/vlm/2025_07_11_17_51_01/results.json", "r") as f:
        #     gen_opts = {}
        #     for line in f.readlines():
        #         item = json.loads(line)
        #         gen_opts[item['QA']['id']] = item['QA']['gen_opts']
        # ############################

        self.question_types = set()

        self.rows = []
        self.question_types_map = {}
        row_id = 0
        for i in range(len(data)):
            video_path = os.path.join(self.video_dir, f"{data[i]['key']}.mp4")
            for j in range(len(data[i]['qa'])):
                opts = data[i]['qa'][j]['question'].split("\n")[1:]
                opts = [o[4:] for o in opts]

                # ############################
                # opts = [o.strip() for o in gen_opts[row_id]]
                # ############################

                ques = data[i]['qa'][j]['question'].split("\n")[0]
                full_prompt = data[i]['qa'][j]['question']
                for qatype in data[i]['qa'][j]['question_type']:
                    self.question_types.add(qatype)
                self.question_types_map[row_id] = data[i]['qa'][j]['question_type']
                self.rows.append({
                    "id" : row_id,
                    "video_path" : video_path,
                    "ques" : ques,
                    "opts" : opts,
                    "gt_ans" : data[i]['qa'][j]['answer'],
                    "full_prompt" : full_prompt + "\n" + "Answer with the option's letter from the given choices directly.\n",
                })
                row_id += 1

    def check_answer(self, ans, gt_ans):
        if len(ans) >= 3 and ans[0] =='(' and ans[2] == ')' and ans[1] == gt_ans[0]:
            return True
        if len(ans) > 0 and ans[0] == gt_ans[0]:
            return True
        return False
            
    def metrics(self, result_path):
        res = {qatype : {"true_count":0,"total_count":0} for qatype in self.question_types}
        res['overall'] = {"true_count":0,"total_count":0}
        with open(result_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                is_true = False
                if self.check_answer(item['ans'],item['QA']['gt_ans']):
                    res['overall']['true_count']+=1
                    is_true = True
                res['overall']['total_count']+=1
                for qatype in self.question_types_map[item['QA']['id']]:
                    res[qatype]['total_count']+=1
                    if is_true:
                        res[qatype]['true_count']+=1
        for qatype in res:
            res[qatype]['accuracy'] = res[qatype]['true_count']/res[qatype]['total_count'] if res[qatype]['total_count']>0 else 0
        return res
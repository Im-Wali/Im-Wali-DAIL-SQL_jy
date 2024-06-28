import argparse
import os
import json

import openai
from tqdm import tqdm

from llm.chatgpt import init_chatgpt, ask_llm
from utils.enums import LLM
from torch.utils.data import DataLoader

from utils.post_process import process_duplication, get_sqls

QUESTION_FILE = "questions.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_group_id", type=str, default="org-ktBefi7n9aK7sZjwc2R9G1Wo")
    parser.add_argument("--model", type=str, choices=[LLM.TEXT_DAVINCI_003, 
                                                      LLM.GPT_35_TURBO,
                                                      LLM.GPT_35_TURBO_0613,
                                                      # LLM.TONG_YI_QIAN_WEN,
                                                      LLM.GPT_35_TURBO_16K,
                                                      LLM.LLAMA3_8B,
                                                      LLM.GPT_4],
                        default=LLM.GPT_35_TURBO)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mini_index_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set")
    parser.add_argument("--db_dir", type=str, default="dataset/spider/database")
    args = parser.parse_args()

    # check args
    assert args.model in LLM.BATCH_FORWARD or \
           args.model not in LLM.BATCH_FORWARD and args.batch_size == 1, \
        f"{args.model} doesn't support batch_size > 1"

    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]

    # init openai api
    # init_chatgpt(args.openai_api_key, args.openai_group_id, args.model)

    if args.start_index == 0:
        mode = "w"
    else:
        mode = "a"

    if args.mini_index_path:
        mini_index = json.load(open(args.mini_index_path, 'r'))
        questions = [questions[i] for i in mini_index]
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}_MINI.txt"
    else:
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}.txt"
    out_all_file = f"{args.question}/RESULTS_MODEL-ALL-{args.model}.txt"    

    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)

    token_cnt = 0
    with open(out_file, mode) as f:
        with open(out_all_file,mode) as f2:
            for i, batch in enumerate(tqdm(question_loader)):
                if i < args.start_index:
                    continue
                if i >= args.end_index:
                    break
                try:
                    res = ask_llm(args.model, batch, args.temperature, args.n)
                    # print(res);
                    # print(res["response"]);
                except openai.error.InvalidRequestError:
                    print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
                    res = ""

                # parse result
                # token_cnt += res["total_tokens"]

                # 'content' 키의 값을 가져옴
                content = res['response']['content']
                f2.write(content)
                content = content.replace("```","") 
                content = content.replace(";","") 
                sql = " ".join(content.replace("\n", " ").split())
                sql = sql + "   " + db_ids[i]
                print(sql)
                if sql.startswith("SELECT"):
                    f.write(sql + "\n")
                elif sql.startswith(" "):
                    f.write("SELECT" + sql + "\n")
                else:
                    f.write("SELECT " + sql + "\n")
                # if args.n == 1:
                #     # for sql in res["response"]:
                #     for sql in sql_query:
                #         # remove \n and extra spaces
                #         sql = " ".join(sql.replace("\n", " ").split())
                #         # sql = process_duplication(sql)
                #         print("SQL1 : " + sql)
                #         # python version should >= 3.8
                #         if sql.startswith("SELECT"):
                #             f.write(sql + "\n")
                #         elif sql.startswith(" "):
                #             f.write("SELECT" + sql + "\n")
                #         else:
                #             f.write("SELECT " + sql + "\n")
                # else:
                #     results = []
                #     cur_db_ids = db_ids[i * args.batch_size: i * args.batch_size + len(batch)]
                #     # for sqls, db_id in zip(res["response"], cur_db_ids):
                #     for sqls, db_id in zip(sql_query, cur_db_ids):
                #         processed_sqls = []
                #         for sql in sqls:
                #             sql = " ".join(sql.replace("\n", " ").split())
                #             # sql = process_duplication(sql)
                #             print("SQL2 : " + db_id)
                #             print("SQL2 : " + sql)
                #             if sql.startswith("SELECT"):
                #                 pass
                #             elif sql.startswith(" "):
                #                 sql = "SELECT" + sql
                #             else:
                #                 sql = "SELECT " + sql
                #             processed_sqls.append(sql)
                #         result = {
                #             'db_id': db_id,
                #             'p_sqls': processed_sqls
                #         }
                #         final_sqls = get_sqls([result], args.n, args.db_dir)

                #         for sql in final_sqls:
                #             f.write(sql + "\n")


import modules.judge_module as judge_module
import argparse
import json
from tqdm import tqdm
import os



def cal_asr(judge_result: list):
    total = len(judge_result)
    success = len([r for r in judge_result if r['result']['is_jailbroken']])
    return success/total



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model_path", type=str, default="openai", help="Path to the judge model, default to OpenAI API")
    parser.add_argument("--data_path", type=str, help="Path to the generated responses")
    parser.add_argument("--save_path", type=str, help="Path to save the result")

    args = parser.parse_args()
    judge_model_path = args.judge_model_path
    data_path = args.data_path
    save_path = args.save_path

    with open(data_path, "r") as f:
        data = json.load(f)
    data = data['results']

    if judge_model_path != "openai":
        judge = judge_module.LlamaGuardJudge(judge_model_path, device_map='auto')
    else:
        judge = judge_module.GPTJudge(openai_api_key=os.environ["OPENAI_API_KEY"])
    print("judge model loaded......")

    total_usage = 0
    judge_result = []
    for i, data_point in tqdm(enumerate(data), total=len(data), desc="Judging"):
        if "no_defense" in data_path or "SemanticSmoothParaphrase" in data_path:
            result, usage = judge.judge(data_point['prompt'], data_point['response'])
            judge_result.append({
                "prompt": data_point['prompt'],
                "response": data_point['response'],
                "result": result
            })
        else:
            result, usage = judge.judge(data_point['rewritten_prompt'], data_point['response'])
            judge_result.append({
                "prompt": data_point['prompt'],
                "rewritten_prompt": data_point['rewritten_prompt'],
                "response": data_point['response'],
                "result": result
            })
        total_usage += usage
        if i == 0:
            print(result)
    
    asr = cal_asr(judge_result)
    print("ASR: ", asr)
    print("Total tokens used: ", total_usage)

    save = {
        "setting": {
            "judge_model_path": judge_model_path,
            "data_path": data_path
        },
        "results": {
            "asr": asr,
            "usage": total_usage,
            "judge_result": judge_result,
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Saving......")
    with open(save_path, "w") as f:
        json.dump(save, f, indent=4)

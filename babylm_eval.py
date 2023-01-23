import argparse
import lm_eval

TASKS = {
    "blimp": ["blimp_determiner_noun_agreement_1",
             "blimp_regular_plural_subject_verb_agreement_1",
             "blimp_wh_island",
             "blimp_passive_1",
             "blimp_npi_present_1"],
    "glue":  ["cola", "sst", "mrpc", "qqp", "mnli", "mnli_mismatched", "qnli", "rte", "wnli"]
}

def accuracy_on_task(task_name, eval_model, template_names):
    eval_task = lm_eval.get_task_list(task_name, template_names=template_names)
    results = lm_eval.evaluate(model=eval_model, tasks=eval_task, seed=12)
    accuracy = results['results'][0]['acc']
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to huggingface model and tokenizer.")
    parser.add_argument("model_type", type=str, choices=["decoder only", "encoder only", "encoder-decoder"],
                        help="Language model architecture.")
    parser.add_argument("--tasks", "-t", type=str, choices=["blimp", "glue", "all"], default="all",
                        help="Tasks on which we evaluate.")
    args = parser.parse_args()

    MODEL_TYPE_REMAP = {"decoder only": "hf-causal",
                        "encoder only": "hf-mlm",
                        "encoder-decoder": "hf-seq2seq",}
    eval_model = lm_eval.get_model(MODEL_TYPE_REMAP[args.model_type],
                                   pretrained=args.model_path,
                                   device="cuda")
    tasks = []
    if args.tasks == "all":
        for task_type in TASKS.keys():
            tasks.extend(TASKS[task_type])
    else:
        tasks = TASKS[args.tasks]

    accuracies = {}
    for task in tasks:
        if task.startswith("blimp"):
            template = "null_prompt"
        else:
            template = lm_eval.list_templates(task)[0]
        accuracies[task] = accuracy_on_task(task, eval_model, [template])

    print("\nScores:")
    for task in tasks:
        print(f"{task}:\t{accuracies[task] * 100:.2f}%")

import argparse
import lm_eval

def accuracy_on_task(task_name, eval_model, template_names):
    eval_task = lm_eval.get_task_list(task_name, template_names=template_names)
    results = lm_eval.evaluate(model=eval_model, tasks=eval_task, seed=12)
    accuracy = results['results'][0]['acc']
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to huggingface model and tokenizer.")
    parser.add_argument("model_type", type=str, choices=["decoder only", "encoder-decoder"],
                        help="Language model architecture.")
    args = parser.parse_args()

    MODEL_TYPE_REMAP = {"decoder only": "hf-causal",
                        "encoder-decoder": "hf-seq2seq"}
    eval_model = lm_eval.get_model(MODEL_TYPE_REMAP[args.model_type],
                                   pretrained=args.model_path,
                                   device="cuda")
    tasks = ["blimp_determiner_noun_agreement_1",
             "blimp_regular_plural_subject_verb_agreement_1",
             "blimp_wh_island",
             "blimp_passive_1",
             "blimp_npi_present_1"]

    accuracies = {}
    for task in tasks:
        accuracies[task] = accuracy_on_task(task, eval_model, ["null_prompt"])

    print("\nScores:")
    for task in tasks:
        print(f"{task}:\t{accuracies[task] * 100:.2f}%")

"""Fine-tune T5 with simpletransformers on filtered MultiCochrane English data from paths."""
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import bootstrap_paths  # noqa: F401
from paths import MULTICOHRANE_EN_FILTERED_R05, OUTPUT_ROOT


def train_model(
    model_type,
    model_name,
    train_data,
    eval_data,
    seq_length=256,
    batch_size=8,
    acc_steps=4,
    epochs=1,
    do_sample=True,
    top_k=None,
    top_p=0.95,
    num_beams=1,
    seed=45
):
    model_args = T5Args()
    model_args.max_seq_length = seq_length
    model_args.max_length = 512
    model_args.train_batch_size = batch_size
    model_args.gradient_accumulation_steps = acc_steps
    model_args.gradient_checkpointing = True
    model_args.eval_batch_size = batch_size
    model_args.num_train_epochs = epochs
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 300
    model_args.use_multiprocessing = False
    model_args.fp16 = False
    model_args.save_steps = -1
    model_args.save_eval_checkpoints = False
    model_args.no_cache = True
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.preprocess_inputs = True
    model_args.optimizer = "AdamW"
    model_args.num_return_sequences = 1
    model_args.lazy_loading = False
    model_args.no_save = True
    model_args.do_sample = do_sample
    model_args.top_k = top_k
    model_args.top_p = top_p
    model_args.num_beams = num_beams
    model_args.save_model_every_epoch = False
    model_args.manual_seed = seed 

    model = T5Model(model_type, model_name, args=model_args)
    model.train_model(train_data, eval_data=eval_data)
    return model    


def save_model(model, save_name):
    model.save_pretrained(save_name)


def training_loop():
    # >>> CONFIGURE YOUR PARAMETERS HERE <<<
    train_file = str(MULTICOHRANE_EN_FILTERED_R05 / "train0.5_en.csv")
    val_file = str(MULTICOHRANE_EN_FILTERED_R05 / "val0.5_en.csv")
    do_sample = True
    top_p = 0.95
    save_name = str(OUTPUT_ROOT / "trained_mt5")
    model_name = "google/mt5-small"
    model_type = "mt5"
    epochs = 5

    training_data = pd.read_csv(train_file)
    validation_data = pd.read_csv(val_file)

    model = train_model(
        model_type,
        model_name,
        training_data,
        validation_data,
        do_sample=do_sample,
        top_p=top_p,
        epochs=epochs
    )

    save_model(model.model, save_name)


if __name__ == "__main__":
    training_loop()

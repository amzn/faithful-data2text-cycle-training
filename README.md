## Faithful Low-Resource Data-to-Text Generation Through Cycle Training

This repository contains the source code for the ACL 2023 paper: [Faithful Low-Resource Data-to-Text Generation through Cycle Training](https://aclanthology.org/2023.acl-long.160.pdf)

## How to Run

### Usage

```
python cycle_training.py [-h] [--config_file CONFIG_FILE]
                         [--output_dir OUTPUT_DIR] [--gpu_device GPU_DEVICE]
                         [--bertscore_gpu_device BERTSCORE_GPU_DEVICE]
                         [--t5_tokenizer T5_TOKENIZER]
                         [--data2text_model DATA2TEXT_MODEL]
                         [--text2data_model TEXT2DATA_MODEL]
                         [--text_file TEXT_FILE] [--data_file DATA_FILE]
                         [--max_input_length MAX_INPUT_LENGTH]
                         [--min_output_length MIN_OUTPUT_LENGTH]
                         [--max_output_length MAX_OUTPUT_LENGTH]
                         [--num_beams NUM_BEAMS]
                         [--no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE]
                         [--length_penalty LENGTH_PENALTY] [--do_train]
                         [--seed SEED] [--num_epochs NUM_EPOCHS]
                         [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                         [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                         [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                         [--data2text_learning_rate DATA2TEXT_LEARNING_RATE]
                         [--text2data_learning_rate TEXT2DATA_LEARNING_RATE]
                         [--scheduler_type SCHEDULER_TYPE]
                         [--warmup_steps WARMUP_STEPS]
                         [--adaptive_type ADAPTIVE_TYPE]
                         [--scorer_model_tokenizer SCORER_MODEL_TOKENIZER]
                         [--scorer_model SCORER_MODEL] [--do_eval]
                         [--data2text_validation_file DATA2TEXT_VALIDATION_FILE]
                         [--text2data_validation_file TEXT2DATA_VALIDATION_FILE]
                         [--do_generate] [--do_test]
                         [--data2text_test_file DATA2TEXT_TEST_FILE]
                         [--text2data_test_file TEXT2DATA_TEST_FILE]
                         [--save_epochs SAVE_EPOCHS]
                         [--selection_metric SELECTION_METRIC] [--delta DELTA]
                         [--patience PATIENCE]
````

### Optional arguments
```
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Optional use of config file for passing the arguments
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written
  --gpu_device GPU_DEVICE
                        GPU device id
  --bertscore_gpu_device BERTSCORE_GPU_DEVICE
                        GPU device id for bertscore model
  --t5_tokenizer T5_TOKENIZER
                        Tokenizer for T5 models
  --data2text_model DATA2TEXT_MODEL
                        Local or Huggingface transformer's path to the
                        data2text model
  --text2data_model TEXT2DATA_MODEL
                        Local or Huggingface transformer's path to the
                        text2data_model model
  --text_file TEXT_FILE
                        Text used for cycle training (text-data-text cycle)
  --data_file DATA_FILE
                        Data used for cycle training (data-text-data cycle)
  --max_input_length MAX_INPUT_LENGTH
                        Maximum input length including prompt after
                        tokenization
  --min_output_length MIN_OUTPUT_LENGTH
                        Minimum output length
  --max_output_length MAX_OUTPUT_LENGTH
                        Maximum output length
  --num_beams NUM_BEAMS
                        Number of beams for beam search
  --no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE
                        No repeat ngram size
  --length_penalty LENGTH_PENALTY
                        Length penalty
  --do_train            Whether to run training.
  --seed SEED           Random seed
  --num_epochs NUM_EPOCHS
                        Total number of training epochs to perform.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass; effective training
                        batch size equals to per_gpu_train_batch_size *
                        gradient_accumulation_steps
  --data2text_learning_rate DATA2TEXT_LEARNING_RATE
                        The initial learning rate of AdamW for the data2text
                        model; larger learning rate is suggested for T5
                        families
  --text2data_learning_rate TEXT2DATA_LEARNING_RATE
                        The initial learning rate of AdamW for the text2data
                        model; larger learning rate is suggested for T5
                        families
  --scheduler_type SCHEDULER_TYPE
                        Learning rate scheduler type (linear/cosine/cosine_wit
                        h_restarts/polynomial/constant/constant_with_warmup)
  --warmup_steps WARMUP_STEPS
                        Scheduler warmup steps
  --adaptive_type ADAPTIVE_TYPE
                        0: No adaptive learning; 1: adaptive instance weighted
                        loss; 2: adaptive learning rate
  --scorer_model_tokenizer SCORER_MODEL_TOKENIZER
                        Tokenizer for the scorer model
  --scorer_model SCORER_MODEL
                        Local path to the scorer model
  --do_eval             Whether to run eval on the dev set
  --data2text_validation_file DATA2TEXT_VALIDATION_FILE
                        The development set of the data2text task
  --text2data_validation_file TEXT2DATA_VALIDATION_FILE
                        The development set of the text2data task
  --do_generate         Whether to run generation for the evaluation of the
                        dev set
  --do_test             Whether to run eval on the test set
  --data2text_test_file DATA2TEXT_TEST_FILE
                        The test set of the data2text task
  --text2data_test_file TEXT2DATA_TEST_FILE
                        The test set of the text2data task
  --save_epochs SAVE_EPOCHS
                        Save model every X updates epochs
  --selection_metric SELECTION_METRIC
                        The metric used for model section; --do_generate
                        required for metric other than loss
  --delta DELTA         Minimum requirement of improvement
  --patience PATIENCE   Terminate the training after n epochs without any
                        improvement
  
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


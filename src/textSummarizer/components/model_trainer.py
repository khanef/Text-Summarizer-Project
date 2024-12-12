from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from textSummarizer.entity import ModelTrainerConfig
import torch

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def check_gpu_memory(self, device):
        """Kiểm tra mức sử dụng bộ nhớ GPU và trả về True nếu vượt ngưỡng 80%."""
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device.index)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_percent = (info.used / info.total) * 100
        print(f"GPU memory usage: {used_percent:.2f}%")
        nvmlShutdown()
        return used_percent >= 80

    def train(self):
        # Kiểm tra và chọn thiết bị
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")


        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        # Load model với xử lý lỗi CUDA
        try:
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        except RuntimeError as e:
            print("CUDA error occurred, switching to CPU...")
            device = torch.device("cpu")
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Data collator
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=500,
            save_steps=1e6,
            gradient_accumulation_steps=16
        )

        # Trainer
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )
        
        # Bắt đầu huấn luyện
        trainer.train()

        # Lưu model và tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

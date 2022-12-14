import json
import string
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import LightningDataModule




class CustomDataset(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.src = df.stt_out.values.tolist()
        self.tgt = df.sentence.values.tolist()
    
    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, index):
        source_text = self.src[index]
        target_text = self.tgt[index]
        sample = {"src": source_text, "tgt": target_text}
        return sample


def collate_batch(batch):
    tokenizer = AutoTokenizer.from_pretrained("../model/opus-mt-nl-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("../model/opus-mt-nl-en")

    en_batch = [x['tgt'] for x in batch]
    en_tgt = tokenizer(en_batch, return_tensors="pt", padding=True).input_ids
    en_tgt = model.generate(en_tgt, num_beams=5, max_new_tokens=512)
    en_tgt = tokenizer.batch_decode(en_tgt, skip_special_tokens=True)

    source = [x['src'].strip().translate(str.maketrans('', '', string.punctuation)) for x in batch]
    target = [x['tgt'].strip().translate(str.maketrans('', '', string.punctuation))  for x in batch]
    interim_target = [x.strip().translate(str.maketrans('', '', string.punctuation))  for x in en_tgt]
    return source, target, interim_target


class STTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "../data/common_voice_nl", batch_size: int = 16, num_workers: int = 6) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _preproc(self, split: str):
        df = pd.read_csv(f'{self.data_dir}/{split}.tsv', sep="\t")
        all_stt = []
        for fname in df.path.values:
            with open(f"{self.data_dir}/cv_nl_stt/{fname}.json", "r+") as f:
                stt_out = json.load(f)
                all_stt.append(stt_out['results']['channels'][0]['alternatives'][0]['transcript'])
        df['stt_out'] = pd.Series(all_stt)
        df['sentence'] = df.sentence.str.lower()
        df['sentence'] = df.sentence.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        return df
    
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: str):
        if stage == "test" or stage is None:
            self.ds_test = CustomDataset(self._preproc("test"))
        if stage == "fit" or stage is None:
            self.ds_val = CustomDataset(self._preproc("dev"))
            self.ds_train = CustomDataset(self._preproc("train"))
    
    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch
        )
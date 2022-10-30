from fnmatch import translate
import json
from os import stat
import string
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from pytorch_lightning import LightningDataModule


tqdm.pandas()


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
    source = [x['src'].strip().translate(str.maketrans('', '', string.punctuation)) for x in batch]
    target = [x['tgt'].strip().translate(str.maketrans('', '', string.punctuation))  for x in batch]

    return source, target


class STTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "../data/common_voice_nl", batch_size: int = 16,
    num_workers: int = 6, device: str = 'cpu') -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
    
    def _translate_nl_en(self, sentence, src_lang, tgt_lang='en_XX'):
        model = MBartForConditionalGeneration.from_pretrained("../model/mbart-large-50-one-to-many-mmt")
        model = model.to(self.device)
        tokenizer = MBart50TokenizerFast.from_pretrained("../model/mbart-large-50-one-to-many-mmt")

        tokenizer.src_lang = src_lang
        encoded = tokenizer(sentence, return_tensors="pt").to(self.device)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_new_tokens=512
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    def _preproc(self, split: str):
        df = pd.read_csv(f'{self.data_dir}/{split}.tsv', sep="\t")
        all_stt = []
        for fname in df.path.values:
            with open(f"{self.data_dir}/cv_nl_stt/{fname}.json", "r+") as f:
                stt_out = json.load(f)
                all_stt.append(stt_out['results']['channels'][0]['alternatives'][0]['transcript'])
        df['stt_out'] = pd.Series(all_stt)
        df['sentence'] = df.sentence.str.lower()
        df['nl_en'] = df.sentence.progress_apply(lambda x: self._translate_nl_en(x, 'nl_XX'))

        df['sentence'] = df.sentence.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df['nl_en'] = df.nl_en.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
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
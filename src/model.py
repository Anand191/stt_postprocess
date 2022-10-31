import string
import torch
from torch import nn
from jiwer import wer, cer
from typing import Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import LightningModule
from loguru import logger


class chained_seq2seq(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer_encoder = AutoTokenizer.from_pretrained("../model/opus-mt-nl-en")
        self.tokenizer_decoder = AutoTokenizer.from_pretrained("../model/opus-mt-en-nl")

        self.encoder = AutoModelForSeq2SeqLM.from_pretrained("../model/opus-mt-nl-en")
        self.decoder = AutoModelForSeq2SeqLM.from_pretrained("../model/opus-mt-en-nl")
        # output is logits. CE loss applies log_softmax to logits and then computes nll loss
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)

    
    def forward(self, input_ids, tgt_ids):
        # logger.info(f"source batch size = {input_ids.size()}")
        outputs_interim = self.encoder.generate(input_ids, num_beams=5, max_new_tokens=512) 
        interim_batch = self.tokenizer_encoder.batch_decode(outputs_interim, skip_special_tokens=True)
        interim_batch = [x.strip().translate(str.maketrans('', '', string.punctuation)) for x in interim_batch]

        decoder_input_ids = self.tokenizer_decoder(interim_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        decoder_attention_masks = self.tokenizer_decoder(interim_batch, return_tensors="pt", padding=True).attention_mask.to(self.device)
        # logger.info(f"decoder batch size = {decoder_input_ids.size()}")
        outputs_final = self.decoder(input_ids=decoder_input_ids, attention_mask = decoder_attention_masks, labels=tgt_ids)

        return outputs_final

    def training_step(self, batch, batch_idx):
        src_batch, tgt_batch = batch
        src_ids = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        tgt_ids = self.tokenizer_encoder(tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)
        outputs = self.forward(src_ids, tgt_ids)
        loss, logits = outputs[:2]
        generated = self.generate(logits)
        word_err = torch.tensor(wer(tgt_batch, generated))
        char_err = torch.tensor(wer(tgt_batch, generated))

        self.log_dict({'train/loss':loss, 'train/wer': word_err, 'train/cer': char_err},
        batch_size=tgt_ids.size()[0], on_step=True, prog_bar=True
        )

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        src_batch, tgt_batch = batch
        src_ids = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        tgt_ids = self.tokenizer_encoder(tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)
        outputs = self.forward(src_ids, tgt_ids)
        loss, logits = outputs[:2]
        generated = self.generate(logits)
        word_err = torch.tensor(wer(tgt_batch, generated))
        char_err = torch.tensor(wer(tgt_batch, generated))

        self.log_dict({'val/loss':loss, 'val/wer': word_err, 'val/cer': char_err},
        batch_size=tgt_ids.size()[0], on_step=False, on_epoch=True, prog_bar=True
        )
    
    def test_step(self, batch, batch_idx):
        src_batch, tgt_batch = batch
        src_ids = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        tgt_ids = self.tokenizer_encoder(tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)
        outputs = self.forward(src_ids, tgt_ids)
        loss, logits = outputs[:2]
        generated = self.generate(logits)
        word_err = torch.tensor(wer(tgt_batch, generated))
        char_err = torch.tensor(wer(tgt_batch, generated))

        self.log_dict({'test/loss':loss, 'test/wer': word_err, 'test/cer': char_err},
        batch_size=tgt_ids.size()[0], on_step=False, on_epoch=True, prog_bar=True
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def generate(self, logits):
        preds = nn.functional.softmax(logits, dim=-1).argmax(dim=-1)
        final_batch = self.tokenizer_decoder.batch_decode(sequences=preds, skip_special_tokens=True)
        return final_batch
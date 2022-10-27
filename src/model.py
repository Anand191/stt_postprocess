import string
import torch
from torch import nn
from jiwer import wer
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)

        # self.example_input_array = self._get_example_array()

    
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
        word_err = torch.tensor(wer(tgt_batch, self.generate(logits)))

        self.log_dict({'train/loss':loss, 'train/wer': word_err},
        batch_size=tgt_ids.size()[0], on_step=True, prog_bar=True
        )

        return {"loss": loss, "wer": word_err}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_wer = torch.stack([x['wer'] for x in outputs]).mean()
        tensorboard_logs = {
            'train/loss': avg_loss,
            'train/wer': avg_wer,
            'step': self.current_epoch
        }
        # return {'loss': avg_loss, 'wer': avg_wer, 'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_idx):
        src_batch, tgt_batch = batch
        src_ids = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        tgt_ids = self.tokenizer_encoder(tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)
        outputs = self.forward(src_ids, tgt_ids)
        loss, logits = outputs[:2]
        word_err = torch.tensor(wer(tgt_batch, self.generate(logits)))

        self.log_dict({'val/loss':loss, 'val/wer': word_err},
        batch_size=tgt_ids.size()[0], on_step=False, on_epoch=True, prog_bar=True
        )

        return {"val_loss": loss, "val_wer": word_err}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_wer = torch.stack([x['val_wer'] for x in outputs]).mean()
        tensorboard_logs = {
            'val/loss': avg_loss,
            'val/wer': avg_wer,
            'step': self.current_epoch
        }
        return {'val/loss': avg_loss, 'val/wer': avg_wer,'log': tensorboard_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    # def _get_example_array(self):
    #     sample_src_batch = ["ik ben "*6 for _ in range(16)]
    #     sample_src_batch = [x.strip().translate(str.maketrans('', '', string.punctuation)) for x in sample_src_batch]
    #     sample_src_ids = self.tokenizer_encoder(sample_src_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
    #     sample_tgt_batch = ["ben ik "*6 for _ in range (16)]
    #     sample_tgt_batch = [x.strip().translate(str.maketrans('', '', string.punctuation)) for x in sample_tgt_batch]
    #     sample_tgt_ids = self.tokenizer_encoder(sample_tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)
    #     return sample_src_ids, sample_tgt_ids

    
    # def _shared_step(self, src_batch, tgt_batch):
    #     tgt_ids = self.tokenizer_encoder(tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)
    #     outputs = self.forward(src_batch, tgt_ids)
    #     loss, logits = outputs[:2]
    #     word_err = wer(tgt_batch, self.generate(logits))
    #     logger.info(f"target batch size = {tgt_ids.size()}")
    #     logger.info(f"model output shape = {logits.size()}")
    #     logger.info(f"returned loss = {loss}, calculated loss={self.criterion(logits[-1, :], tgt_ids[-1, :])}")
    #     logger.info(f"loss = {loss}; wer = {word_err}")
    #     self.log('loss', loss, batch_size=tgt_ids.size()[0])
    #     self.log('wer', word_err, batch_size=tgt_ids.size()[0])
    #     return loss

    def generate(self, logits):
        preds = nn.functional.softmax(logits, dim=-1).argmax(dim=-1)
        final_batch = self.tokenizer_decoder.batch_decode(sequences=preds, skip_special_tokens=True)
        return final_batch
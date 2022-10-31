import string
import torch
from collections import namedtuple
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

        self.example_input_array = self._get_example_array()

    
    def forward(self, input_ids=None, interim_attn_masks=None, interim_ids=None, tgt_ids=None):
        outputs_interim = self.encoder(input_ids=input_ids, attention_mask = interim_attn_masks, labels=interim_ids)
        _, mid_logits = outputs_interim[:2]

        middle_batch = self.generate(mid_logits)
        middle_batch = [x.strip().translate(str.maketrans('', '', string.punctuation)) for x in middle_batch]
        # logger.info(f"middle_batch = {middle_batch}")

        decoder_input_ids = self.tokenizer_decoder(middle_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        decoder_attention_masks = self.tokenizer_decoder(middle_batch, return_tensors="pt", padding=True).attention_mask.to(self.device)
        outputs_final = self.decoder(input_ids=decoder_input_ids, attention_mask = decoder_attention_masks, labels=tgt_ids)

        return outputs_interim, outputs_final

    def training_step(self, batch, batch_idx):
        src_batch, tgt_batch, interim_batch = batch

        src_ids = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        interim_attention_masks = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).attention_mask.to(self.device)
        interim_ids = self.tokenizer_decoder(interim_batch, return_tensors="pt", padding=True).input_ids.to(self.device)        
        tgt_ids = self.tokenizer_encoder(tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)

        payload = {
            'input_ids': src_ids,
            'interim_attn_masks': interim_attention_masks, 
            'interim_ids': interim_ids, 
            'tgt_ids': tgt_ids
        }
        out_interim, out_final = self.forward(**payload)

        loss_interim, logits_interim = out_interim[:2]
        loss_final, logits_final = out_final[:2]
        word_err = torch.tensor(wer(tgt_batch, self.generate(logits_final)))

        self.log_dict({'train/loss':loss_interim+loss_final, 'train/wer': word_err},
        batch_size=tgt_ids.size()[0], on_step=True, prog_bar=True
        )

        return {"loss": loss_interim+loss_final, "wer": word_err}

    def validation_step(self, batch, batch_idx):
        src_batch, tgt_batch, interim_batch = batch

        src_ids = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).input_ids.to(self.device)
        interim_attention_masks = self.tokenizer_encoder(src_batch, return_tensors="pt", padding=True).attention_mask.to(self.device)
        interim_ids = self.tokenizer_decoder(interim_batch, return_tensors="pt", padding=True).input_ids.to(self.device)        
        tgt_ids = self.tokenizer_encoder(tgt_batch, return_tensors="pt",padding=True).input_ids.to(self.device)

        payload = {
            'input_ids': src_ids,
            'interim_attn_masks': interim_attention_masks, 
            'interim_ids': interim_ids, 
            'tgt_ids': tgt_ids
        }
        out_interim, out_final = self.forward(**payload)

        loss_interim, logits_interim = out_interim[:2]
        loss_final, logits_final = out_final[:2]
        word_err = torch.tensor(wer(tgt_batch, self.generate(logits_final)))

        self.log_dict({'val/loss':loss_interim+loss_final, 'val/wer': word_err},
        batch_size=tgt_ids.size()[0], on_step=False, on_epoch=True, prog_bar=True
        )

        # return {"val_loss": loss, "val_wer": word_err}
    
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_wer = torch.stack([x['val_wer'] for x in outputs]).mean()
    #     tensorboard_logs = {
    #         'val/loss': avg_loss,
    #         'val/wer': avg_wer,
    #         'step': self.current_epoch
    #     }
    #     return {'val/loss': avg_loss, 'val/wer': avg_wer,'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def generate(self, logits):
        preds = nn.functional.softmax(logits, dim=-1).argmax(dim=-1)
        final_batch = self.tokenizer_decoder.batch_decode(sequences=preds, skip_special_tokens=True)
        return final_batch
    
    def _get_example_array(self):
        sample_src_ids = torch.randint(1, 10, (16, 23)).to(self.device)
        sample_interim_ids = torch.randint(1, 10, (16, 21)).to(self.device)
        sample_interim_mask = torch.randint(1, 10, (16, 23)).to(self.device)
        sample_tgt_ids = torch.randint(1, 10, (16, 26)).to(self.device)
        packaged = {
            'input_ids': sample_src_ids,
            'interim_attn_masks': sample_interim_mask, 
            'interim_ids': sample_interim_ids, 
            'tgt_ids': sample_tgt_ids
        }
        return packaged
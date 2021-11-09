'''
The work in this article is based on Facebook's work to be modified.
'''

import torch
from torch.nn import functional as F
import math
import os
import sys
#from pytorch_transformers import *
from transformers import *

from wsd_models.util import *

def load_projection(path):
    proj_path = os.path.join(path, 'best_probe.ckpt')
    with open(proj_path, 'rb') as f: proj_layer = torch.load(f)
    return proj_layer

class PretrainedClassifier(torch.nn.Module):
    def __init__(self, num_labels, encoder_name, proj_ckpt_path):
        super(PretrainedClassifier, self).__init__()

        self.encoder, self.encoder_hdim = load_pretrained_model(encoder_name)

        if proj_ckpt_path and len(proj_ckpt_path) > 0:
            self.proj_layer = load_projection(proj_ckpt_path)
            #assert to make sure correct dims
            assert self.proj_layer.in_features == self.encoder_hdim
            assert self.proj_layer.out_features == num_labels
        else:
            self.proj_layer = torch.nn.Linear(self.encoder_hdim, num_labels)

    def forward(self, input_ids, input_mask, example_mask):
        output = self.encoder(input_ids, attention_mask=input_mask)[0]

        example_arr = []        
        for i in range(output.size(0)): 
            example_arr.append(process_encoder_outputs(output[i], example_mask[i], as_tensor=True))
        output = torch.cat(example_arr, dim=0)
        output = self.proj_layer(output)
        return output



class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss

        self.linear = torch.nn.Linear(768, 768)

    def forward(self, input_ids, attn_mask, output_mask=None):

        #encode gloss text
        if self.is_frozen:
            with torch.no_grad(): 
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]


        if output_mask is None:
            #training model to put all sense information on CLS token 
            gloss_output = gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        else:
            #average representations over target word(s)
            example_arr = []        
            for i in range(gloss_output.size(0)): 
                example_arr.append(process_encoder_outputs2(gloss_output[i], output_mask[i], as_tensor=True))
            gloss_output = torch.cat(example_arr, dim=0)


        return self.linear(gloss_output)



class ExampleEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(ExampleEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = True#freeze_gloss

    def forward(self, input_ids, attn_mask, output_mask=None):

        #encode gloss text
        if self.is_frozen:
            with torch.no_grad():
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]


        if output_mask is None:
            #training model to put all sense information on CLS token 
            gloss_output = gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        else:
            #average representations over target word(s)
            example_arr = []
            for i in range(gloss_output.size(0)):
                example_arr.append(process_encoder_outputs2(gloss_output[i], output_mask[i], as_tensor=True))
            gloss_output = torch.cat(example_arr, dim=0)

        return gloss_output







class GlossEncoder2(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss

    def forward(self, input_ids, attn_mask):

        #encode gloss text
        if self.is_frozen:
            with torch.no_grad(): 
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        #training model to put all sense information on CLS token 
        gloss_output = gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        return gloss_output

class GlossEncoder3(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss
        self.rnn = torch.nn.RNN(input_size=768, hidden_size=768, num_layers=1)


    def forward(self, input_ids, attn_mask):
        #encode gloss text
        if self.is_frozen:
            with torch.no_grad(): 
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        #training model to put all sense information on CLS token 
        gloss_output = gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        return gloss_output




class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context):
        super(ContextEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context

        self.linear = torch.nn.Linear(768, 768)


    def forward(self, input_ids, attn_mask, output_mask, flag):
        #encode context
        if self.is_frozen:
            with torch.no_grad(): 
                context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]

        #average representations over target word(s)
        example_arr = []        
        for i in range(context_output.size(0)): 
            if flag: example_arr.append(process_encoder_outputs2(context_output[i], output_mask[i], as_tensor=True))
            else: example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)


        return self.linear(context_output)




class ContextEncoder2(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context):
        super(ContextEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim, self.embeddings = load_pretrained_model2(encoder_name)
        self.is_frozen = freeze_context
        

    def forward(self, input_ids, attn_mask, output_mask):
        #encode context
        if self.is_frozen:
            with torch.no_grad(): 
                context_output = self.context_encoder(inputs_embeds=input_ids, attention_mask=attn_mask)[0]
                #context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_output = self.context_encoder(inputs_embeds=input_ids, attention_mask=attn_mask)[0]
            #context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]

        #average representations over target word(s)
        example_arr = []        
        for i in range(context_output.size(0)): 
            example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)

        return context_output




class Encoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze):
        super(Encoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.encoder, self.hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze

    def forward(self, input_ids, attn_mask, output_mask=None):
        #encode context
        if self.is_frozen:
            with torch.no_grad(): 
                output = self.encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            output = self.encoder(input_ids, attention_mask=attn_mask)[0]

        if output_mask is None:
            #training model to put all sense information on CLS token 
            output = output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        else:
            #average representations over target word(s)
            example_arr = []        
            for i in range(output.size(0)): 
                example_arr.append(process_encoder_outputs(output[i], output_mask[i], as_tensor=True))
            output = torch.cat(example_arr, dim=0)

        return output




class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, tie_encoders=False):
        super(BiEncoderModel, self).__init__()

        #tying encoders for ablation
        self.tie_encoders = tie_encoders

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = ContextEncoder(encoder_name, freeze_context)
        if self.tie_encoders:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)
            #self.example_encoder = ExampleEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)
        else:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss)
            #self.example_encoder = ExampleEncoder(encoder_name, freeze_gloss)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim

    def context_forward(self, context_input, context_input_mask, context_example_mask, flag=False):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask, flag)

    def gloss_forward(self, gloss_input, gloss_mask, output_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask, output_mask)

    #def example_forward(self, gloss_input, gloss_mask, output_mask):
    #    return self.example_encoder.forward(gloss_input, gloss_mask, output_mask)


#EOF

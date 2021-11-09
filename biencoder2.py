import os, gc, sys, time, math, copy, argparse, pickle, random

import torch
from torch.nn import functional as F

import numpy as np
from tqdm import tqdm
from transformers import *

import nltk
from nltk.corpus import wordnet as wn
import nltk.stem; st = nltk.stem.SnowballStemmer('english')

from wsd_models.util import *
from wsd_models.models import BiEncoderModel, Encoder



'''Parameter parser''' 
parser = argparse.ArgumentParser(description='Quantum-inspired Representation Method for WSD')

'''common arguments'''
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--silent', action='store_true',
	help='Flag to supress training progress bar for each epoch')
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--lr', type=float, default=0.00001) #0.00001
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--context-max-length', type=int, default=128)
parser.add_argument('--gloss-max-length', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--context-bsz', type=int, default=4) #default 4, by me change to 1
parser.add_argument('--gloss-bsz', type=int, default=256)
parser.add_argument('--encoder-name', type=str, default='bert-base',
	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
parser.add_argument('--ckpt', type=str, required=True,
	help='filepath at which to save best probing model (on dev set)')
parser.add_argument('--data-path', type=str, required=True,
	help='Location of top-level directory for the Unified WSD Framework')

'''sets which parts of the model to freeze during training for ablation'''
parser.add_argument('--freeze-context', action='store_true')
parser.add_argument('--freeze-gloss', action='store_true')
parser.add_argument('--tie-encoders', action='store_true')

'''other training settings flags'''
parser.add_argument('--kshot', type=int, default=-1,
	help='if set to k (1+), will filter training data to only have up to k examples per sense')
parser.add_argument('--balanced', action='store_true',
	help='flag for whether or not to reweight sense losses to be balanced wrt the target word')

'''evaluation arguments'''
parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')
parser.add_argument('--split', type=str, default='semeval2007',
	choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'all-test'],
	help='Which evaluation split on which to evaluate probe')




'''uses these two gpus if training in multi-gpu'''
context_device, gloss_device = "cuda:0", "cuda:1"



def tokenize_glosses(gloss_arr, tokenizer, max_len):
	glosses, masks = [], []

	for gloss_text in gloss_arr:
		g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)+tokenizer.encode(gloss_text, add_special_tokens=False)+tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)]
		g_attn_mask = [1]*len(g_ids)
		g_fake_mask = [-1]*len(g_ids)
		g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])
		g_ids = torch.cat(g_ids, dim=-1)
		g_attn_mask = torch.tensor(g_attn_mask)
		glosses.append(g_ids)
		masks.append(g_attn_mask)

	return glosses, masks






def tokenize_examples(example_arr, tokenizer, max_len, lemma, pos):
	examples, masks, outputs = [], [], []

	pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}
	for example_text in example_arr:

		e_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index
		o_masks, flag = [-1], -1

		#For each word in sentence...
		for idx, word in enumerate(nltk.word_tokenize(example_text.lower())):
			#tensorize word for context ids
			word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower(), add_special_tokens=False)]
			e_ids.extend(word_ids)

			#if word is labeled with WSD sense...
			if st.stem(word) == st.stem(lemma) or nltk.stem.WordNetLemmatizer().lemmatize(word, 'v') == st.stem(lemma) :
				#add word to bert output mask to be labeled
				o_masks.extend([idx]*len(word_ids))
				flag += 1

			else:
				#mask out output of context encoder for WSD task (not labeled)
				o_masks.extend([-1]*len(word_ids))

			#break if we reach max len
			if max_len != -1 and len(e_ids) >= (max_len-1):
				break

		e_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token
		e_attn_masks = [1]*len(e_ids)
		o_masks.append(-1)
		if flag == -1: o_masks[0] = 0 #没有找到
		
		e_ids, e_attn_masks, o_masks = normalize_length(e_ids, e_attn_masks, o_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])

		
		e_ids        = torch.cat(e_ids, dim=-1)
		e_attn_masks = torch.tensor(e_attn_masks)
		o_masks      = torch.tensor(o_masks)


		examples.append(e_ids)
		masks.append(e_attn_masks)
		outputs.append(o_masks)


		'''
		g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(gloss_text)]
		g_attn_mask = [1]*len(g_ids)
		g_fake_mask = [-1]*len(g_ids)

		g_ids, g_attn_mask, g_fake_mask = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])
		g_ids = torch.cat(g_ids, dim=-1)

		

		flag = -1
		#for i, w in enumerate(tokenizer.tokenize(gloss_text)):
		for i, w in enumerate(nltk.word_tokenize(gloss_text.lower())):
			if nltk.stem.WordNetLemmatizer().lemmatize(w, pos_converter[pos]) is lemma:
				g_fake_mask[i+1] = i + 1
				flag = i + 1
				continue
		if flag == -1: g_fake_mask[0] = 0 #没有找到

		g_attn_mask = torch.tensor(g_attn_mask)
		g_fake_mask = torch.tensor(g_fake_mask)
		'''

			
	return examples, masks, outputs





def tokenize_examples_old(gloss_arr, tokenizer, max_len, lemma, pos):
	glosses, masks, outputs = [], [], []

	pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}
	for gloss_text in gloss_arr:

		g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(gloss_text)]
		g_attn_mask = [1]*len(g_ids)
		g_fake_mask = [-1]*len(g_ids)

		g_ids, g_attn_mask, g_fake_mask = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])
		g_ids = torch.cat(g_ids, dim=-1)

		
		flag = -1
		#for i, w in enumerate(tokenizer.tokenize(gloss_text)):
		for i, w in enumerate(nltk.word_tokenize(gloss_text.lower())):
			if nltk.stem.WordNetLemmatizer().lemmatize(w, pos_converter[pos]) is lemma:
				g_fake_mask[i+1] = i + 1
				flag = i + 1
				continue
		if flag == -1: g_fake_mask[0] = 0 #没有找到

			
		g_attn_mask = torch.tensor(g_attn_mask)
		g_fake_mask = torch.tensor(g_fake_mask)

		glosses.append(g_ids)
		masks.append(g_attn_mask)
		outputs.append(g_fake_mask)

	return glosses, masks, outputs
	


def load_and_preprocess_example_sentences(data, tokenizer, wn_senses, max_len=-1):
	sense_examples, sense_weights, examples_lengths   = {}, {}, []

	for sent in data:

		for _, lemma, pos, _, label in sent:
			if label == -1:
				continue #ignore unlabeled words
			else:
				key = generate_key(lemma, pos)
				if key not in sense_examples:
					#get all sensekeys for the lemma/pos pair
					sensekey_arr = wn_senses[key]


					# 只选第一个示例
					#examples_arr = [wn.lemma_from_key(s).synset().examples()[0] for s in sensekey_arr]
					examples_arr = []
					for s in sensekey_arr:

                                            
						definition = wn.lemma_from_key(s).synset().definition()
						if definition is None or len(definition) == 0:
							print('....')

						else:
							if len(wn.lemma_from_key(s).synset().examples()) > 0:
								examples_arr.append(wn.lemma_from_key(s).synset().examples()[0])
							else:
								#如果没有找到就给定义
								examples_arr.append(definition)
								#print(">>> Don't have examples")



					example_ids, example_masks, example_outp_masks = tokenize_examples(examples_arr, tokenizer, max_len, lemma, pos)
					example_ids = torch.cat(example_ids, dim=0)
					example_masks = torch.stack(example_masks, dim=0)
					example_outp_masks = torch.stack(example_outp_masks, dim=0)
					sense_examples[key] = (example_ids, example_masks, example_outp_masks, sensekey_arr)

					#intialize weights for balancing senses
					sense_weights[key] = [0]*len(examples_arr)
					w_idx = sensekey_arr.index(label)
					sense_weights[key][w_idx] += 1
				else:
					#update sense weight counts
					w_idx = sense_examples[key][3].index(label)
					sense_weights[key][w_idx] += 1
				
				#make sure that gold label is retrieved synset
				assert label in sense_examples[key][3]

	#normalize weights
	for key in sense_weights:
		total_w = sum(sense_weights[key])
		sense_weights[key] = torch.FloatTensor([total_w/x if x !=0 else 0 for x in sense_weights[key]])

	return sense_examples, sense_weights









'''creates a sense label/ gloss dictionary for training/using the gloss encoder'''
def load_and_preprocess_glosses(data, tokenizer, wn_senses, max_len=-1):
	sense_glosses, sense_weights, gloss_lengths = {}, {}, []

	for sent in data:
		for _, lemma, pos, _, label in sent:
			if label == -1:
				continue #ignore unlabeled words
			else:
				key = generate_key(lemma, pos)

				if key not in sense_glosses:
					#get all sensekeys for the lemma/pos pair
					sensekey_arr = wn_senses[key]
					#get glosses for all candidate senses
					gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]


					#preprocess glosses into tensors
					gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
					gloss_ids = torch.cat(gloss_ids, dim=0)
					gloss_masks = torch.stack(gloss_masks, dim=0)
					sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr)

					#intialize weights for balancing senses
					sense_weights[key] = [0]*len(gloss_arr)
					w_idx = sensekey_arr.index(label)
					sense_weights[key][w_idx] += 1
				else:
					#update sense weight counts
					w_idx = sense_glosses[key][2].index(label)
					sense_weights[key][w_idx] += 1
				
				#make sure that gold label is retrieved synset
				assert label in sense_glosses[key][2]

	#normalize weights
	for key in sense_weights:
		total_w = sum(sense_weights[key])
		sense_weights[key] = torch.FloatTensor([total_w/x if x !=0 else 0 for x in sense_weights[key]])

	return sense_glosses, sense_weights



def preprocess_context(tokenizer, text_data, bsz=1, max_len=-1):
	if max_len == -1: assert bsz==1 #otherwise need max_length for padding

	context_ids, context_attn_masks, example_keys = [], [], []
	context_output_masks, instances, labels = [], [], []

	#tensorize data
	for sent in text_data:
		c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index
		o_masks, sent_insts, sent_keys, sent_labels = [-1], [], [], []

		#For each word in sentence...
		for idx, (word, lemma, pos, inst, label) in enumerate(sent):
			#tensorize word for context ids
			word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower(), add_special_tokens=False)]
			c_ids.extend(word_ids)

			#if word is labeled with WSD sense...
			if inst != -1:
				#add word to bert output mask to be labeled
				o_masks.extend([idx]*len(word_ids))
				#track example instance id
				sent_insts.append(inst)
				#track example instance keys to get glosses
				ex_key = generate_key(lemma, pos)
				sent_keys.append(ex_key)
				sent_labels.append(label)
			else:
				#mask out output of context encoder for WSD task (not labeled)
				o_masks.extend([-1]*len(word_ids))

			#break if we reach max len
			if max_len != -1 and len(c_ids) >= (max_len-1):
				break

		c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token
		c_attn_mask = [1]*len(c_ids)
		o_masks.append(-1)
		c_ids, c_attn_masks, o_masks = normalize_length(c_ids, c_attn_mask, o_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])

		y = torch.tensor([1]*len(sent_insts), dtype=torch.float)
		#not including examples sentences with no annotated sense data
		if len(sent_insts) > 0:
			context_ids.append(torch.cat(c_ids, dim=-1))
			context_attn_masks.append(torch.tensor(c_attn_masks).unsqueeze(dim=0))
			context_output_masks.append(torch.tensor(o_masks).unsqueeze(dim=0))
			example_keys.append(sent_keys)
			instances.append(sent_insts)
			labels.append(sent_labels)

	#package data
	data = list(zip(context_ids, context_attn_masks, context_output_masks, example_keys, instances, labels))

	#batch data if bsz > 1
	if bsz > 1:
		print('Batching data with bsz={}...'.format(bsz))
		batched_data = []
		for idx in range(0, len(data), bsz):
			if idx+bsz <=len(data): b = data[idx:idx+bsz]
			else: b = data[idx:]
			context_ids = torch.cat([x for x,_,_,_,_,_ in b], dim=0)
			context_attn_mask = torch.cat([x for _,x,_,_,_,_ in b], dim=0)
			context_output_mask = torch.cat([x for _,_,x,_,_,_ in b], dim=0)
			example_keys = []
			for _,_,_,x,_,_ in b: example_keys.extend(x)
			instances = []
			for _,_,_,_,x,_ in b: instances.extend(x)
			labels = []
			for _,_,_,_,_,x in b: labels.extend(x)
			batched_data.append((context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels))
		return batched_data
	else:  
		return data






def _train(train_data, model, linear, train_gloss_dict, optim, schedule, criterion, train_example_dict, gloss_bsz=-1, max_grad_norm=1.0, multigpu=False, silent=False, train_steps=-1):
	model.train()
	total_loss = 0.

	start_time = time.time()

	train_data = enumerate(train_data)
	if not silent: train_data = tqdm(list(train_data))

	for i, (context_ids, context_attn_mask, context_output_mask, example_keys, _, labels) in train_data:

		'''reset model'''
		model.zero_grad()
		
		'''run example sentence(s) through context encoder'''
		#if multigpu:
		#	context_ids = context_ids.to(context_device)
		#	context_attn_mask = context_attn_mask.to(context_device)
		#	context_output_mask = context_output_mask.to(context_device)
		#else:
		#	context_ids = context_ids.cuda()
		#	context_attn_mask = context_attn_mask.cuda()
		#	context_output_mask = context_output_mask.cuda()
		context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)
		




		loss, gloss_sz, context_sz = 0., 0, len(labels)
		for j, (key, label) in enumerate(zip(example_keys, labels)):

			output = context_output.split(1,dim=0)[j]


			

			"""Added by Junwei Zhang in 20210930"""
			output1 = output / torch.sqrt(torch.sum(torch.mul(output, output), 1)).squeeze()
			output2 = torch.flip(output1, dims=[1])
			PA = linear(output)
			PA = PA.squeeze(0)

			output_quantum = output1 * torch.cos(PA) + output2 * torch.sin(PA)



		


			'''run example's glosses through gloss encoder'''
			gloss_ids, gloss_attn_mask, sense_keys = train_gloss_dict[key]
		
			#if multigpu:
			#	gloss_ids = gloss_ids.to(gloss_device)
			#	gloss_attn_mask = gloss_attn_mask.to(gloss_device)
			#else:
			#	gloss_ids = gloss_ids.cuda()
			#	gloss_attn_mask = gloss_attn_mask.cuda()
			gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask, None)
			gloss_output = gloss_output.transpose(0,1)



			"""Added by Junwei Zhang in 20210930"""
			gloss_output_quantum = gloss_output / torch.sqrt(torch.sum(torch.mul(gloss_output, gloss_output), 0)).unsqueeze(0)






			'''Added by Junwei Zhang'''
			example_ids, exmaple_attn_mask, example_outp_mask, _ = train_example_dict[key]
			#if multigpu:
			#	example_ids = example_ids.to(context_device)
			#	exmaple_attn_mask = exmaple_attn_mask.to(context_device)
			#	example_outp_mask = example_outp_mask.to(context_device)
			#else:
			#	example_ids = example_ids.cuda()
			#	exmaple_attn_mask = exmaple_attn_mask.cuda()
			#	example_outp_mask = example_outp_mask.cuda()
			example_output = model.context_forward(example_ids, exmaple_attn_mask, example_outp_mask, flag=True)
			example_output = example_output.transpose(0,1)



			"""Added by Junwei Zhang in 20210930"""
			example_output_quantum = example_output / torch.sqrt(torch.sum(torch.mul(example_output, example_output), 0)).unsqueeze(0)



			#if multigpu:
			#	gloss_output = gloss_output.to(gloss_device)
			#	example_output = example_output.to(gloss_device)
			#	output = output.to(gloss_device)
			#	gloss_output_quantum = gloss_output_quantum.to(gloss_device)
			#	example_output_quantum = example_output_quantum.to(gloss_device)
			#	output_quantum = output_quantum.to(gloss_device)
				
			#else:
			#	gloss_output = gloss_output.cuda()
			#	example_output = example_output.cuda()
			#	output = output.cuda()
			#	gloss_output_quantum = gloss_output_quantum.cuda()
			#	example_output_quantum = example_output_quantum.cuda()
			#	output_quantum = output_quantum.cuda()
				






			"""Added by Junwei Zhang in 20210930"""
			if gloss_output.size() != example_output.size(): print(':> size of gloss_output not equal to example_output ...')
			output_1 = torch.mm(output, gloss_output)
			output_2 = torch.mm(output, example_output)
			output_3 = torch.mm(output_quantum, gloss_output_quantum)
			output_4 = torch.mm(output_quantum, example_output_quantum)

			result = output_1 + output_2 + output_3 + output_4


			

			'''get label and calculate loss'''
			idx = sense_keys.index(label)
			label_tensor = torch.tensor([idx])




			#if multigpu:
			#	label_tensor = label_tensor.cuda()
			#	result = result.cuda()
			#else:
			#	label_tensor = label_tensor.cuda()
			#	result = result.cuda()



			'''looks up correct candidate senses criterion'''
			'''needed if balancing classes within the candidate senses of a target word'''
			loss += criterion[key](result, label_tensor)
			gloss_sz += gloss_output.size(-1)

			if gloss_bsz != -1 and gloss_sz >= gloss_bsz:
				'''update model'''
				total_loss += loss.item()
				loss=loss/gloss_sz
				loss.backward()#retain_graph=True
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optim.step()
				schedule.step() # Update learning rate schedule

				'''reset loss and gloss_sz'''
				loss = 0.
				gloss_sz = 0

				'''reset model'''
				model.zero_grad()

				'''rerun context through model'''
				context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

		'''update model after finishing context batch'''
		if gloss_bsz != -1: loss_sz = gloss_sz
		else: loss_sz = context_sz
		if loss_sz > 0:
			total_loss += loss.item()
			loss=loss/loss_sz
			loss.backward() #retain_graph=True
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optim.step()
			schedule.step() # Update learning rate schedule

		'''stop epoch early if number of training steps is reached'''
		if train_steps > 0 and i+1 == train_steps: break

	return model, optim, schedule, total_loss







def _eval(eval_data, model, linear, gloss_dict, example_dict, multigpu=False):
	model.eval()
	eval_preds = []
	for context_ids, context_attn_mask, context_output_mask, example_keys, insts, _ in eval_data:
		with torch.no_grad(): 


			'''run example through model'''
			if multigpu:
				context_ids         = context_ids.to(context_device)
				context_attn_mask   = context_attn_mask.to(context_device)
				context_output_mask = context_output_mask.to(context_device)
			else:
				context_ids         = context_ids.cuda()
				context_attn_mask   = context_attn_mask.cuda()
				context_output_mask = context_output_mask.cuda()

			context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)




			for output, key, inst in zip(context_output.split(1,dim=0), example_keys, insts):


				"""Added by Junwei Zhang in 20210930"""
				output1 = output / torch.sqrt(torch.sum(torch.mul(output, output), 1)).squeeze()
				output2 = torch.flip(output1, dims=[1])
				PA = linear(output)
				PA = PA.squeeze(0)

				output_quantum = output1 * torch.cos(PA) + output2 * torch.sin(PA)




				'''run example's glosses through gloss encoder'''
				gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]

				if multigpu:
					gloss_ids = gloss_ids.to(gloss_device)
					gloss_attn_mask = gloss_attn_mask.to(gloss_device)
				else:
					gloss_ids = gloss_ids.cuda()
					gloss_attn_mask = gloss_attn_mask.cuda()


				gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask, None)
				gloss_output = gloss_output.transpose(0,1)

				"""Added by Junwei Zhang in 20210930"""
				gloss_output_quantum = gloss_output / torch.sqrt(torch.sum(torch.mul(gloss_output, gloss_output), 0)).unsqueeze(0)



				"""Added by Junwei Zhang"""
				example_ids, exmaple_attn_mask, example_outp_mask, _ = example_dict[key]
				if multigpu:
					example_ids = example_ids.to(context_device)
					exmaple_attn_mask = exmaple_attn_mask.to(context_device)
					example_outp_mask = example_outp_mask.to(context_device)
				else:
					example_ids = example_ids.cuda()
					exmaple_attn_mask = exmaple_attn_mask.cuda()
					example_outp_mask = example_outp_mask.cuda()
				example_output = model.context_forward(example_ids, exmaple_attn_mask, example_outp_mask, flag=True)
				example_output = example_output.transpose(0,1)

				"""Added by Junwei Zhang in 20210930"""
				example_output_quantum = example_output / torch.sqrt(torch.sum(torch.mul(example_output, example_output), 0)).unsqueeze(0)


				#if multigpu:
				#	gloss_output   = gloss_output.to(gloss_device)
				#	example_output = example_output.to(gloss_device)
				#	output         = output.to(gloss_device)
				#	gloss_output_quantum   = gloss_output_quantum.to(gloss_device)
				#	example_output_quantum = example_output_quantum.to(gloss_device)
				#	output_quantum         = output_quantum.to(gloss_device)
				#else:
				#	gloss_output   = gloss_output.cuda()
				#	example_output = example_output.cuda()
				#	output         = output.cuda()
				#	gloss_output_quantum   = gloss_output_quantum.cuda()
				#	example_output_quantum = example_output_quantum.cuda()
				#	output_quantum         = output_quantum.cuda()






				"""Added by Junwei Zhang in 20210930"""
				if gloss_output.size() != example_output.size(): print(':> size of gloss_output not equal to example_output ...')
				output_1 = torch.mm(output, gloss_output)
				output_2 = torch.mm(output, example_output)
				output_3 = torch.mm(output_quantum, gloss_output_quantum)
				output_4 = torch.mm(output_quantum, example_output_quantum)
				result   = output_1 + output_2 + output_3 + output_4




				pred_idx = result.topk(1, dim=-1)[1].squeeze().item()
				pred_label = sense_keys[pred_idx]
				eval_preds.append((inst, pred_label))

	return eval_preds





def train_model(args):

	print('Training WSD model...')

	'''no gloss bsz if not training gloss encoder, memory concerns'''
	if args.freeze_gloss: assert args.gloss_bsz == -1 

	'''create passed in ckpt dir if doesn't exist'''
	if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

	'''LOAD PRETRAINED TOKENIZER, TRAIN AND DEV DATA'''
	print('Loading data + preprocessing...')
	sys.stdout.flush()

	tokenizer = load_tokenizer(args.encoder_name)


	'''loading WSD (semcor) data'''
	train_path = os.path.join(args.data_path, 'Training_Corpora/SemCor/')
	train_data = load_data(train_path, 'semcor')

	'''filter train data for k-shot learning'''
	if args.kshot > 0: train_data = filter_k_examples(train_data, args.kshot)

	'''dev set = semeval2007'''
	semeval2007_path = os.path.join(args.data_path, 'Evaluation_Datasets/semeval2007/')
	semeval2007_data = load_data(semeval2007_path, 'semeval2007')

	'''load gloss dictionary (all senses from wordnet for each lemma/pos pair that occur in data)'''
	wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
	wn_senses = load_wn_senses(wn_path)



	'''preprocess and batch data of glosses of NetWord'''
	train_gloss_dict, train_gloss_weights = load_and_preprocess_glosses(train_data, tokenizer, wn_senses, max_len=args.gloss_max_length)
	semeval2007_gloss_dict, _             = load_and_preprocess_glosses(semeval2007_data, tokenizer, wn_senses, max_len=args.gloss_max_length)

	'''preprocess and batch data of examples of NetWord'''
	train_example_dict, train_example_weights = load_and_preprocess_example_sentences(train_data, tokenizer, wn_senses, max_len=args.gloss_max_length)
	semeval2007_example_dict, _               = load_and_preprocess_example_sentences(semeval2007_data, tokenizer, wn_senses, max_len=args.gloss_max_length)
	

	'''preprocess and batch data (context + glosses)'''
	train_data       = preprocess_context(tokenizer, train_data, bsz=args.context_bsz, max_len=args.context_max_length)
	semeval2007_data = preprocess_context(tokenizer, semeval2007_data, bsz=1, max_len=args.context_max_length)




	epochs = args.epochs
	overflow_steps = -1
	t_total = len(train_data)*epochs

	'''if few-shot training, override epochs to calculate num. epochs + steps for equal training signal'''
	if args.kshot > 0:
		'''hard-coded num. of steps of fair kshot evaluation against full model on default numer of epochs'''
		NUM_STEPS = 181500 #num batches in full train data (9075) * 20 epochs 
		num_batches = len(train_data)
		epochs = NUM_STEPS//num_batches #recalculate number of epochs
		overflow_steps = NUM_STEPS%num_batches #num steps in last overflow epoch (if there is one, otherwise 0)
		t_total = NUM_STEPS #manually set number of steps for lr schedule
		if overflow_steps > 0: epochs+=1 #add extra epoch for overflow steps
		print('Overriding args.epochs and training for {} epochs...'.format(epochs))



	''' SET UP FINETUNING MODEL, OPTIMIZER, AND LR SCHEDULE'''
	model  = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context, tie_encoders=args.tie_encoders)
	linear = torch.nn.Linear(768, 1)





	
	'''added by Junwei Zhang'''
	#model_path = os.path.join(args.ckpt, 'best_model.ckpt')
	#model.load_state_dict(torch.load(model_path))
	#linear.load_state_dict(torch.load(model_path))


	'''speeding up training by putting two encoders on seperate gpus (instead of data parallel)'''
	#if args.multigpu: 
	#	model.gloss_encoder = model.gloss_encoder.to(gloss_device)
	#	model.context_encoder = model.context_encoder.to(context_device)
	#	linear = linear.to(context_device)
	#else:
	#	model = model.cuda()
	#	linear = linear.cuda()


	criterion = {}
	if args.balanced:
		for key in train_gloss_dict:
			criterion[key] = torch.nn.CrossEntropyLoss(reduction='none', weight=train_gloss_weights[key])
	else:
		for key in train_gloss_dict:
			criterion[key] = torch.nn.CrossEntropyLoss(reduction='none')

	'''optimize + scheduler from pytorch_transformers package'''
	weight_decay = 0.0 #this could be a parameter
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	adam_epsilon = 1e-8
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon)
	'''schedule = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup, t_total=t_total)'''
	schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup, num_training_steps = t_total)


	'''TRAIN MODEL ON SEMCOR DATA'''
	print('Training probe...')
	best_dev_f1 = .0
	sys.stdout.flush()

	for epoch in range(1, epochs+1):

		'''if last epoch, pass in overflow steps to stop epoch early'''
		train_steps = -1
		if epoch == epochs and overflow_steps > 0: train_steps = overflow_steps

		'''train model for one epoch or given number of training steps'''
		model, optimizer, schedule, train_loss = _train(train_data, model, linear, train_gloss_dict, optimizer, schedule, criterion, train_example_dict, gloss_bsz=args.gloss_bsz, max_grad_norm=args.grad_norm, silent=args.silent, multigpu=args.multigpu, train_steps=train_steps)

		'''eval model on dev set (semeval2007)'''
		eval_preds = _eval(semeval2007_data, model, linear, semeval2007_gloss_dict, semeval2007_example_dict, multigpu=args.multigpu)

		'''generate predictions file'''
		pred_filepath = os.path.join(args.ckpt, 'tmp_predictions.txt')
		with open(pred_filepath, 'w') as f:
			for inst, prediction in eval_preds:
				f.write('{} {}\n'.format(inst, prediction))

		'''run predictions through scorer'''
		gold_filepath = os.path.join(args.data_path, 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt')
		scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
		_, _, dev_f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
		print('Dev f1 after {} epochs = {}'.format(epoch, dev_f1))
		sys.stdout.flush() 

		if dev_f1 >= best_dev_f1:
			print('updating best model at epoch {}...'.format(epoch))
			sys.stdout.flush() 
			best_dev_f1 = dev_f1
			'''save to file if best probe so far on dev set'''
			model_fname = os.path.join(args.ckpt, 'best_model.ckpt')
			with open(model_fname, 'wb') as f:
				torch.save(model.state_dict(), f)
				torch.save(linear.state_dict(), f)
			sys.stdout.flush()

		'''shuffle train set ordering after every epoch'''
		random.shuffle(train_data)

		'''clearing cuda added by Junwei Zhang'''
		#gc.collect()
		#torch.cuda.empty_cache()

	return



def evaluate_model(args):
	print('Evaluating WSD model on {}...'.format(args.split))

	'''
	LOAD TRAINED MODEL
	'''
	model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context)
	linear = torch.nn.Linear(768, 1)

	model_path = os.path.join(args.ckpt, 'best_model.ckpt')
	model.load_state_dict(torch.load(model_path))
	linear.load_state_dict(torch.load(model_path))


	model  = model.cuda()
	linear = linear.cuda()

	

	'''
	LOAD TOKENIZER
	'''
	tokenizer = load_tokenizer(args.encoder_name)

	'''
	LOAD EVAL SET
	'''
	eval_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/'.format(args.split))
	eval_data = load_data(eval_path, args.split)

	#load gloss dictionary (all senses from wordnet for each lemma/pos pair that occur in data)
	wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
	wn_senses = load_wn_senses(wn_path)

	gloss_dict, _   = load_and_preprocess_glosses(eval_data, tokenizer, wn_senses, max_len=args.gloss_max_length)
	example_dict, _ = load_and_preprocess_example_sentences(eval_data, tokenizer, wn_senses, max_len=args.gloss_max_length)


	eval_data = preprocess_context(tokenizer, eval_data, bsz=1, max_len=-1)

	'''
	EVALUATE MODEL
	'''
	eval_preds = _eval(eval_data, model, linear, gloss_dict, example_dict, multigpu=False)

	#generate predictions file
	pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(args.split))
	with open(pred_filepath, 'w') as f:
		for inst, prediction in eval_preds:
			f.write('{} {}\n'.format(inst, prediction))

	#run predictions through scorer
	
	'''Classification test added by Junwei Zhang'''
	#gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format('ADV'))
	gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
	scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
	p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
	print('f1 of BERT probe on {} test set = {}'.format(args.split, f1))

	return



if __name__ == "__main__":
	#if not torch.cuda.is_available():
	#	print("Need available GPU(s) to run this model...")
	#	quit()


	'''parse args'''
	args = parser.parse_args()
	print(args)

	'''set random seeds'''
	torch.manual_seed(args.rand_seed)
	os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
	torch.cuda.manual_seed(args.rand_seed)
	torch.cuda.manual_seed_all(args.rand_seed)   
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic=True

	'''evaluate model saved at checkpoint or train model'''
	if args.eval: evaluate_model(args)
	else: train_model(args)



#EOF

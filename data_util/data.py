#Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/data.py

import json
import glob
import random
import struct
import csv
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
       # if len(pieces) != 2:
         # print (f'Warning: incorrectly formatted line in vocabulary file: {line}\n')
         # continue
        w = pieces[0]
        # print('Adding %s to vocabulary file' % w)
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print (f"max_size of vocab was specified as {max_size}; we now have {self._count} words. Stopping reading.")
          break

    print (f"Finished constructing vocabulary of {self._count} total words. Last word added: {self._id_to_word[self._count-1]}")

  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    return self._count

  def write_metadata(self, fpath):
    print (f"Writing word embedding metadata file to {fpath}...")
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in range(self.size()):
        writer.writerow({"word": self._id_to_word[i]})

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[value]))

def example_generator(data_path, single_pass, validids=None):

    filename = data_path
    linecount = 1
    with open(filename) as file_in:
        for line in file_in:
            if linecount not in validids:
                linecount += 1
                continue
            linecount += 1
    
            linearr = json.loads(line.strip())
            uid = linearr[0]
            question = linearr[1]
            intermediate_sparql = linearr[8]
            if not question or not intermediate_sparql:
                continue

            question = question.replace('?', ' ?') #.replace('{','').replace('}','')
            intermediate_sparql = intermediate_sparql.replace('vr0.','vr0 .').replace('vr1.','vr1 .').replace('COUNT(?','COUNT ( ?').replace('vr0)','vr0 )').replace('vr1)','vr1 )').replace('(?','( ?')
                                 
            questiontokens = linearr[2]
            questionvectors = linearr[3]
            ents = linearr[4]
            rels = linearr[5]
            #finents = linearr[6]
            #finrels = linearr[7]
        

            # enc_input = questionvectors[:max_enc_len]
            # enc_len = len(enc_input)
            # if enc_len == 0:
            #    continue
            # question_words = [w.lower() for w in questiontokens][:max_enc_len]
            #enc_len = len(question_words)
            # enc_input_mask = [vocab.word_to_id(w) for w in question_words]
            # enc_input_extend_vocab, question_oovs = Data_Helper.article_to_ids(question_words, vocab)

            for idx,ent in enumerate(ents):
                intermediate_sparql = intermediate_sparql.replace(ent,'entpos@@'+str(ents.index(ent)+1))
            for idx,rel in enumerate(rels):
                intermediate_sparql = intermediate_sparql.replace(rel,'predpos@@'+str(rels.index(rel)+1))
            # sparqladd = ' [sep] ' + ' '.join(ents) + ' [sep] ' + ' '.join(rels)
            # intermediate_sparql += sparqladd
                    
            output = {
            
                        'article' : question,
                        'abstract': intermediate_sparql,
                        'questokens': questiontokens,
                        'quesvectors': questionvectors,
                        'ents': ents,
                        'rels':rels,
                        'uid': uid
            }

            yield output
            '''
            dict = {
                'article': bytes(question, 'utf-8'),
                'abstract': bytes(intermediate_sparql, 'utf-8')
            }


            print("QUES VEC: ", len(questionvectors))
            print("11111: ", len(questionvectors[0]))
            output = example_pb2.Example(features=feature_pb2.Features(feature={
                'article': _bytes_feature(dict['article']),
                'abstract': _bytes_feature(dict['abstract']),
                'quesvectors': _float_feature(questionvectors)
             }))
            yield output
            '''

    if single_pass:
        print("example_generator completed reading all datafiles. No more data.")
    return

def article2ids(article_words, vocab):
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in article_words:
    
    if len(w) > 2:
        if w[0] == "'" and w[-1] == "'" : #sparql specific hack for handling questions like "which cities start with the letter L "
            w = w[1:-1]
            # print("a2i singlequote: ",article_words,w)

        if w[0] == '"' and w[-1] == '"':
            w = w[1:-1]
            # print("a2i doubequote: ",article_words,w)
    
                        
        try:                             #to handle questions like "which buildings have a height larger than 200.50 meters"
            w = str(float(w))
            # print("Article a2i float: ",article_words,w)            
        except ValueError:
            pass

    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs
        oovs.append(w)
      oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
      ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
      ids.append(i)

  return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    
    if len(w) > 2:
        if w[0] == "'" and w[-1] == "'" : #sparql specific hack for handling questions like "which cities start with the letter L "
            w = w[1:-1]
           # print("a2i singlequote: ",abstract_words,w)

        if w[0] == '"' and w[-1] == '"':
            w = w[1:-1]
            #print("a2i doubequote: ",abstract_words,w)

                        
        try:                             #to handle questions like "which buildings have a height larger than 200.50 meters"
            w = str(float(w))
            # print("Abstract a2i float: ",abstract_words,w)            
        except ValueError:
            pass

    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in article_oovs: # If w is an in-article OOV
        vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-article OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, article_oovs):
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) # might be [UNK]
    except ValueError as e: # w is OOV
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.size()
      if article_oov_idx < len(article_oovs):
          w = article_oovs[article_oov_idx]
      #try:
      #  w = article_oovs[article_oov_idx]
      #except ValueError as e: # i doesn't correspond to an article oov
      #  raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
      else:
          print("Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs" % (i, article_oov_idx, len(article_oovs)))
          continue
    words.append(w)
  return words


def abstract2sents(abstract):
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: # no more sentences
      return sents


def show_art_oovs(article, vocab):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = article.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = abstract.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: # w is oov
      if article_oovs is None: # baseline mode
        new_words.append("__%s__" % w)
      else: # pointer-generator mode
        if w in article_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else: # w is in-vocab word
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str

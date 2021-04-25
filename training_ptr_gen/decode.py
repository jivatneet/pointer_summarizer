#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division
import importlib

import sys
import json

importlib.reload(sys)
#in py3 is hard-wired to "utf-8"
#sys.setdefaultencoding('utf8')

import os
import time
from fuzzywuzzy import fuzz

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from train_util import get_input_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()
out = []

class Beam(object):
  def __init__(self, tokens, log_probs, state, state1, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.state1 = state1
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, state1, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      state1 = state1,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        qcount = 0
        totalfuzz = 0.0

        while batch is not None:
            res = {}
            # Run beam search to get best Hypothesis
            best_summaries = self.beam_search(batch)
            question = batch.questions[0]
            uid = batch.uid[0]
            ents = batch.ents[0]
            rels = batch.rels[0]
            res['uid'] = uid
            res['question'] = question
            res['goldents'] = ents
            res['goldrels'] = rels

            for idx, best_summary in enumerate(best_summaries):
                
                # Extract the output ids from the hypothesis and convert back to words
                output_ids = [int(t) for t in best_summary.tokens[1:]]
                decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))
                
                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words

                original_abstract_sents = batch.original_abstracts_sents[0]

                # target, answer = write_for_rouge(original_abstract_sents, decoded_words, counter,
                #                  self._rouge_ref_dir, self._rouge_dec_dir)

                target = original_abstract_sents
                answer = ' '.join(decoded_words)
                res['target'] = target
                res['answer_{}'.format(idx)] = answer
                print('answer_{}: {}'.format(idx, answer))

            if res['target'].split() == res['answer_0'].split():
                res['exact_match'] = 'match'
            else:
                res['exact_match'] = 'no match'

            out.append(res)

            qcount += 1
            totalfuzz += fuzz.ratio(target.lower(), res['answer_0'].lower())
            print("target: ", target)
            #print("answer: ", answer,'\n')
            print("avg fuzz after %d questions = %f"%(qcount,float(totalfuzz)/qcount))
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        #results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        #rouge_log(results_dict, self._decode_dir)


    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens, enc_padding_mask)
        s_t_0 = self.model.reduce_state(encoder_hidden, decode =True)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        # dec_h = dec_h.squeeze()
        # dec_c = dec_c.squeeze()
        dec_h0 = dec_h[0].squeeze()
        dec_h1 = dec_h[1].squeeze()
        dec_c0 = dec_c[0].squeeze()
        dec_c1 = dec_c[1].squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h0[0], dec_c0[0]),
                      state1 = (dec_h1[0], dec_c1[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h = []
            all_state_c = []

            all_context = []

            all_state_h1 = []
            all_state_c1 = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

                state_h1, state_c1 = h.state1
                all_state_h1.append(state_h1)
                all_state_c1.append(state_c1)

            hcat = torch.cat((torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_h1, 0).unsqueeze(0)), dim=0)
            ccat = torch.cat((torch.stack(all_state_c, 0).unsqueeze(0), torch.stack(all_state_c1, 0).unsqueeze(0)), dim=0)
            s_t_1 = (hcat, ccat)
            c_t_1 = torch.stack(all_context, 0)


            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)
    
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h0 = dec_h[0].squeeze()
            dec_h1 = dec_h[1].squeeze()
            dec_c0 = dec_c[0].squeeze()
            dec_c1 = dec_c[1].squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h0[i], dec_c0[i])
                state_i1 = (dec_h1[i], dec_c1[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   state1 = state_i1,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[:]

if __name__ == '__main__':
    model_filename = sys.argv[1]
    file_out = open(sys.argv[2], 'w')
    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode()
    file_out.write(json.dumps(out, indent=4, sort_keys=False))
    file_out.close()



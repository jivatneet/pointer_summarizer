from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import glob

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad
from torch.autograd import Variable

from fuzzywuzzy import fuzz

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from data_util.utils import write_for_rouge
from train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

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

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        
        self.testbatcher = Batcher(config.eval_data_path, self.vocab, mode='decode',
                              batch_size=config.beam_size, single_pass=False)
        
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter, fuzz = False, fuzzval = None):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }

        if fuzz:
            model_save_path = os.path.join(self.model_dir, 'model_fuzz_%d_%d_%f' % (iter, int(time.time()), fuzzval))

        else:
            model_save_path = os.path.join(self.model_dir, 'model_loss_%d_%d_%f' % (iter, int(time.time()), running_avg_loss))
        torch.save(state, model_save_path)

#-----------------------------------------DECODING-------------------------------------------------------------------------#
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):

        print("Starting validation")
        batch = self.testbatcher.next_batch()
        counter = 0
        qcount = 0
        totalfuzz = 0.0

        while qcount < 100:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

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

            target = original_abstract_sents
            answer = ' '.join(decoded_words)
            # target, answer = write_for_rouge(original_abstract_sents, decoded_words, counter,
            #                                 None, None)

            qcount += 1
            totalfuzz += fuzz.ratio(target.lower(), answer.lower())
            print("target: ", target)
            print("answer: ", answer, '\n')
            print("avg fuzz after %d questions = %f" % (qcount, float(totalfuzz) / qcount))
            
            counter += 1
            batch = self.testbatcher.next_batch()

        print("Ending validation")
        return totalfuzz/100.0, qcount

    def beam_search(self, batch):
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens, enc_padding_mask)
        # print("VAL HIDDEN LEN: ", len(encoder_hidden))
        # print("VAL HIDDEN: ", encoder_hidden[0].shape)
        s_t_0 = self.model.reduce_state(encoder_hidden, decode=True)
        # print("VAL HIDDEN AFTER: ", s_t_0[0].shape)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
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

            # print("ALL S 1st ele: ", all_state_h[0].shape, all_state_h1[0].shape)
            hcat = torch.cat((torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_h1, 0).unsqueeze(0)), dim=0)
            ccat = torch.cat((torch.stack(all_state_c, 0).unsqueeze(0), torch.stack(all_state_c1, 0).unsqueeze(0)), dim=0)
            s_t_1 = (hcat, ccat)

            # print("ST111111: ", s_t_1[0].shape)

            # s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            # print("DECODE ST1: ", s_t_1[0].shape)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                                                    encoder_outputs, encoder_feature,
                                                                                    enc_padding_mask, c_t_1,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t_1, steps)
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

        return beams_sorted[0]

    #-----------------------------------------------------------------------------------------------------------------------#
    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def ls_combine(x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y    
        
    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        # print('Enc batch in train', enc_batch)
        # print('Target batch in train', target_batch)
        self.model.encoder = self.model.encoder.train()
        self.model.decoder = self.model.decoder.train()
        self.model.reduce_state = self.model.reduce_state.train()
        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens, enc_padding_mask)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)

            # print('TEST OUTPUT\n', testoutput)
            # print('TEST SHAPE: ', testoutput.shape)
            # print('TARGET BATCH\n', target_batch)
            # print('TARGET SHAPE: ', target_batch.shape)
            # for answer,target in zip(testoutput,target_batch):
                           # qcount += 1
#                        #print("target: ",[x for x in list(target.numpy()) if x != 1 and x!=3])
#                        #print("answer: ",[x for x in list(tf.math.argmax(answer, axis=1).numpy()) if x != 1 and x!= 3])
                           # for x in list(target.cpu().numpy()):
                            #    print(self.vocab.id2word(x))
                            #   print(self.vocab.id2word(x).decode('utf-8'))

                           # target = ' '.join([self.vocab.id2word(x).decode('utf-8') for x in list(target.cpu().numpy()) if x != 1 and x != 3])
                           # answer = ' '.join([self.vocab.id2word(x).decode('utf-8') for x in list(torch.argmax(answer.cpu().detach()).numpy()) if x != 1 and x!= 3])
            
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)

            # label smoothing
            '''
            log_probs = -torch.log(final_dist + config.eps)
            loss = log_probs.sum(dim = 1)
            K = final_dist.size()[-1]
            loss_k = loss / K
            epsilon = config.ls_eps
            step_loss = epsilon * loss_k + (1 - epsilon) * step_loss
            # step_loss = self.ls_combine(loss_k, step_loss, config.ls_eps)
            '''

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        best_fuzz = 0.0
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
                print('steps %d loss: %f running avg loss: %f' % (iter, loss, running_avg_loss))

            # perform validation
            if iter%100 == 0 and iter > 1:
                self.model.encoder = self.model.encoder.eval()
                self.model.decoder = self.model.decoder.eval()
                self.model.reduce_state = self.model.reduce_state.eval()

                with torch.no_grad():
                    valfuzz, qcount = self.decode()
                
                    # write valfuzz to summary_writer
                    summary = tf.Summary()
                    tag_name = 'data/val_fuzz'
                    summary.value.add(tag=tag_name, simple_value=valfuzz)
                    self.summary_writer.add_summary(summary, iter)

                    # update best valiation set accuracy
                    if valfuzz > best_fuzz:
                        print("Updating best model. Earlier fuzz: {} New best: {}".format(best_fuzz, valfuzz))
                        best_fuzz = valfuzz

                        snapshot_prefix = os.path.join(self.model_dir, 'best_snapshot')
                        snapshot_path = snapshot_prefix + '_loss_{}_fuzz_{}'.format(running_avg_loss, best_fuzz)

                        state = {
                             'iter': iter,
                             'encoder_state_dict': self.model.encoder.state_dict(),
                             'decoder_state_dict': self.model.decoder.state_dict(),
                             'reduce_state_dict': self.model.reduce_state.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'current_loss': running_avg_loss
                         }

                        # save model, delete previous 'best_snapshot' files
                        torch.save(state, snapshot_path)
                        for f in glob.glob(snapshot_prefix + '*'):
                            if f != snapshot_path:
                                os.remove(f)

            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 1000 == 0:
                self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)

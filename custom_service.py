import os
import mxnet as mx
import numpy as np
import rnn_model
import bisect

from mxnet import profiler

class CustomService(object):

    def __init__(self):
        self.model = None
        self.vocab = None
        self.num_lstm_layer = 3
        self.num_embed = 256
        self.num_hidden = 512
        self.initialized = False
    def read_content(self, path):
        with open(path) as ins:
            return ins.read()

    def build_vocab(self, path):
        content = list(self.read_content(path))
        idx = 1
        the_vocab = {}
        for word in content:
            if len(word) == 0:
                continue
            if not word in the_vocab:
                the_vocab[word] = idx
                idx += 1
        return the_vocab

    def initialize(self, context):
            _, arg_params, _ = mx.model.load_checkpoint("/tmp/obama", 75)
            self.vocab = self.build_vocab(os.path.join("/tmp", 'obama.txt'))
            # build an inference model
            self.model = rnn_model.LSTMInferenceModel(
                self.num_lstm_layer,
                len(self.vocab),
                num_hidden=self.num_hidden,
                num_embed=self.num_embed,
                num_label=len(self.vocab),
                arg_params=arg_params,
                ctx=context,
                dropout=0.2)
            self.initialized = True
            return self.vocab

    def MakeRevertVocab(self, vocab):
        dic = {}
        for k, v in vocab.items():
            dic[v] = k
        return dic


    # make input from char
    def MakeInput(self, char, vocab, arr):
        idx = vocab[char]
        tmp = np.zeros((1,))
        tmp[0] = idx
        arr[:] = tmp


    # helper function for random sample
    def _cdf(self, weights):
        total = sum(weights)
        result = []
        cumsum = 0
        for w in weights:
            cumsum += w
            result.append(cumsum / total)
        return result


    def _choice(self, population, weights):
        assert len(population) == len(weights)
        cdf_vals = self._cdf(weights)
        x = np.random.random()
        idx = bisect.bisect(cdf_vals, x)
        return population[idx]


    # we can use random output or fixed output by choosing largest probability
    def MakeOutput(self, prob, vocab, sample=False, temperature=1.):
        if not sample:
            idx = np.argmax(prob, axis=1)[0]
        else:
            fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
            scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
            rescale = np.exp(np.log(scale_prob) / temperature)
            rescale[:] /= rescale.sum()
            return self._choice(fix_dict, rescale[0, :])
        try:
            char = vocab[idx]
        except:
            char = ''
        return char

    def preprocess(self, data):
        # input_data = data[0].get("data")
        # if input_data is None:
        #     input_data = data[0].get("body")
        #
        # # Convert a string of sentence to a list of string
        # sent = input_data[0]["input_sentence"].lower()lower

        return data
        
    def handle(self, data, context):
        seq_length = 1200
        # profiler.set_config(profile_all=True, aggregate_stats=True, filename=os.path.join("model",'trace.json'))
        input_ndarray = mx.nd.zeros((1,))
        revert_vocab = self.MakeRevertVocab(self.vocab)
        output = self.preprocess(data)
        random_sample = False
        new_sentence = True
        vocab = self.vocab
        ignore_length = len(output)
        for i in range(seq_length):
            if i <= ignore_length - 1:
                self.MakeInput(output[i], vocab, input_ndarray)
            else:
                self.MakeInput(output[-1], vocab, input_ndarray)
            # profiler.set_state('run')
            import time

            prob = self.model.forward(input_ndarray, new_sentence)
            mx.nd.waitall()
            # profiler.set_state('stop')
            # profiler.dump()
            new_sentence = False
            next_char = self.MakeOutput(prob, revert_vocab, random_sample)
            if next_char == '':
                new_sentence = True
            if i >= ignore_length - 1:
                output += next_char
        # with open(os.path.join("model", "trace.txt"),'w') as f:
        #     print(profiler.dumps(), f)

        print (output)
        return [{'output': output}]

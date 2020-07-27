import seaborn
import torch  # torch library
import torch.nn as nn  # Nueral Network에 대한 package
import numpy as np  # numpy
import editdistance  # 평가 지표로서 사용될 edit distance
import matplotlib.pyplot as plt  # plot 을 찍기 위한 라이브러리
import tqdm
import torch.nn.functional as F  # pytorch function 들을 사용하기 위한 용도
from torch.utils import data  # dataset 관련된 utility 를 사용하려는 용도
from random import choice, randrange  # random
from itertools import zip_longest
import librosa
import os  # directory 생성 및 디렉토리 생성과 관련된 package
import json
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EOS_TOKEN = '</s>'

# edit distance 란 편집 거리


ref = [1, 2, 3, 4]
hyp = [1, 2, 4, 5, 6]
editdistance.eval(ref, hyp)


def batch(iterable, n=1):
    args = [iter(iterable)] * n
    return zip_longest(*args)


def pad_tensor(vec, pad, value=0, dim=0):
    """
    pad token으로 채우는 용도
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def pad_collate(batch, values=(0, 0), dim=0):
    """
    데이터 로더에 들어가기전에 batch화 할 때 거치는 함수
    args:
        batch - list of (tensor, label)
    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
        ws - a tensor of sequence lengths
    """

    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])  # 각 batch 마다 길이를 얻어내고
    sequence_lengths, xids = sequence_lengths.sort(descending=True)  # 감소하는 순서로 정렬
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    # find longest sequence (가장 긴 sequence의 길이를 구함 )
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    # pad according to max_len (max length 만큼 padd를 추가 )
    batch = [(pad_tensor(x, pad=src_max_len, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for
             (x, y) in batch]

    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.stack([x[1] for x in batch], dim=0)
    xs = xs[xids].contiguous()  # decreasing order로 다시 나열
    ys = ys[xids].contiguous()  # xids 와 같은 순서로
    target_lengths = target_lengths[xids]
    return xs.long(), ys.long(), sequence_lengths.int(), target_lengths.int()


class ToyDataset(data.Dataset):
    """
    https://talbaumel.github.io/blog/attention/
    """

    def __init__(self, min_length=5, max_length=20, type='train'):
        self.SOS = "<s>"  # all strings will end with the End Of String token )
        self.EOS = "</s>"  # all strings will end with the End Of String token
        self.characters = list("abcdefg")
        self.int2char = list(self.characters)
        self.char2int = {c: i + 3 for i, c in enumerate(self.characters)}  # +3 을 왜하는 가?
        print(self.char2int)
        self.VOCAB_SIZE = len(self.characters)
        self.min_length = min_length
        self.max_length = max_length

        # train set or test set 을 생성
        if type == 'train':
            self.set = [self._sample() for _ in range(4000)]
        else:
            self.set = [self._sample() for _ in range(300)]

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

    def _sample(self):
        random_length = randrange(self.min_length, self.max_length)  # Pick a random length
        random_char_list = [choice(self.characters[:-1]) for _ in
                            range(random_length)]  # Pick random chars
        random_string = ''.join(random_char_list)
        a = np.array([self.char2int.get(x) for x in random_string] + [2])
        b = np.array([self.char2int.get(x) for x in random_string[::-1]] + [
            2])  # Return the random string and its reverse + EOS

        return a, b


def check_size(tensor, *args):
    size = [a for a in args]
    assert tensor.size() == torch.Size(size), tensor.size()


def to_mono(y):
    assert y.ndim == 2
    return np.mean(y, axis=1)


def edit_distance(guess, truth):
    guess = guess.split(EOS_TOKEN)[0]
    truth = truth[3:].split(EOS_TOKEN)[0]
    return editdistance.eval(guess, truth) / len(truth)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


## 추후에 설명 Decoder section
def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)  # length 체크
    max_idx = max(seq_len)  # max length 체크
    for n, idx in enumerate(seq_len):  # length 에서 의미없는 hidden state attention 값은 0으로 두기 위한 mask값 설정
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(
                    inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(
                    inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.input_size = config["n_channels"]
        self.hidden_size = config["encoder_hidden"]
        self.layers = config.get("encoder_layers", 1)

        self.dropout = config.get("encoder_dropout", 0.)
        self.bi = config.get("bidirectional_encoder", False)
        embedding_dim = config.get("embedding_dim", None)
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        self.embedding = nn.Embedding(config.get("n_classes", 32), self.embedding_dim,
                                      padding_idx=0)
        gru_input_dim = self.embedding_dim
        self.rnn = nn.GRU(
            gru_input_dim,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)  # model 선언
        self.gpu = config.get("gpu", False)

    def forward(self, inputs, hidden, input_lengths):

        # pack padded 를 통하여 input을 감싸기
        inputs = self.embedding(inputs)

        x = pack_padded_sequence(inputs, input_lengths, batch_first=True)
        output, state = self.rnn(x, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        padding_value=0.)  # sequence 를 위의 그림과 같이 pack함

        if self.bi:  # bidirectional 의 경우 forward와 backward를 sum하여 사용한다. or concat
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
            state = state[:1] + state[1:]
        return output, state

    def init_hidden(self, batch_size):
        # hidden state가 없는 초기 상태일때
        h0 = torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size)
        if self.gpu:
            h0 = h0.cuda()
        return h0


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.batch_size = config["batch_size"]
        self.hidden_size = config["decoder_hidden"]
        embedding_dim = config.get("embedding_dim", None)
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        self.embedding = nn.Embedding(config.get("n_classes", 32), self.embedding_dim,
                                      padding_idx=0)
        self.rnn = nn.GRU(
            input_size=self.embedding_dim + self.hidden_size if config[
                                                                    'decoder'].lower() == 'bahdanau' else self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=config.get("decoder_layers", 1),
            dropout=config.get("decoder_dropout", 0),
            bidirectional=False,
            batch_first=True)
        if config['decoder'] != "RNN":
            self.attention = Attention(
                self.batch_size,
                self.hidden_size,
                method=config.get("attention_score", "dot"))

        self.gpu = config.get("gpu", False)
        self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None

    def forward(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError


class BahdanauDecoder(Decoder):
    """
        Corresponds to BahdanauAttnDecoderRNN in Pytorch tutorial
    """

    def __init__(self, config):
        super(BahdanauDecoder, self).__init__(config)
        self.output_size = config.get("n_classes", 32)
        self.character_distribution = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, **kwargs):
        """
        :param input: [B]
        :param prev_context: [B, H]
        :param prev_hidden: [B, H]
        :param encoder_outputs: [B, T, H]
        :return: output (B), context (B, H), prev_hidden (B, H), weights (B, T)
        """

        input = kwargs["input"]  # decoder input
        prev_hidden = kwargs["prev_hidden"]  # decoder rnn 에서 들어갈 previous hidden state
        encoder_outputs = kwargs["encoder_outputs"]  # encoder RNN에서 Encoding이 끝난 (B,L,hidden_size)
        seq_len = kwargs.get("seq_len", None)  # sequence length

        # check inputs

        # Attention weights
        weights = self.attention.forward(prev_hidden, encoder_outputs, seq_len)  # B x T
        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x H]

        # embed characters
        embedded = self.embedding(input).unsqueeze(0)

        # attention 을 통해 얻어낸 context를 추가하여 모델에 input으로 제공
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), 2)

        outputs, hidden = self.rnn(rnn_input.transpose(1, 0),
                                   prev_hidden.unsqueeze(0))  # B x 1 x H, 1 X B x H

        output = self.character_distribution(outputs.squeeze(0))  # logit 값 각 chracter 별로

        if self.decoder_output_fn:
            # NLL loss 인 경우
            output = self.decoder_output_fn(output, -1)

        if len(output.size()) == 3:
            output = output.squeeze(1)

        return output, hidden.squeeze(0), weights


class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """

    def __init__(self, batch_size, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            # Wa (hidden,hidden)
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            # Wa : (2*hidden,hidden)
            # Va : (hidden,1)
            self.Wa = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        elif method == 'bahdanau':
            # Wa : (hidden_size,hidden_size)
            # Ua : (hidden_size,hidden_size)
            # Va : (hidden_size,1)
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        else:
            raise NotImplementedError

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        """
        Inputs :
          last_hidden : (B,T,hidden_size)
          encoder_outputs :
          seq_len:
        Returns:
          attention matrix :
        """
        batch_size, seq_lens, _ = encoder_outputs.size()
        # attention energies 를 구하기
        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        if seq_len is not None:
            attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`)
        :return:
        """

        # assert last_hidden.size() == torch.Size([batch_size, self.hidden_size]), last_hidden.size()

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)  # (batch_size, hidden_dim,1)

            # attention : (batch_size,max_time, hidden_dim) , (batch_size,hidden_dim,1) - > (batch_size,max_time ,1)

            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            # dot 이랑 비슷 다만 last hidden을 한번 projection
            x = self.Wa(last_hidden)  # (batch_size, hidden_dim) ->  (batch_size, hidden_dim)
            x = x.unsqueeze(-1)  # (batch_size, hidden_dim) ->  (batch_size, hidden_dim,1)
            # encoded 된 hidden states 와 dot proudct를 수행하기
            # attention: (batch_size,max_time, hidden_dim) , (batch_size,hidden_dim,1) - > (batch_size,max_time ,1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1).expand_as(
                encoder_outputs)  # (batch_size, hidden_dim) ->  (batch_size,1, hidden_dim)
            # concat 후 -> linear 거치기 -> 후 tanh
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs),
                                         -1)))  # (batch_size, max_timestep, hidden_dim) ->  (batch_size,  max_timestep, hidden_dim*2)
            # (batch_size, max_timestep, hidden_dim*2) ->  (batch_size,  max_timestep, )
            return x.matmul(self.va).squeeze(-1)

        elif method == "bahdanau":
            # mlp 기반의 attention model

            x = last_hidden.unsqueeze(1)  # (batch_size, hidden_dim) ->  (batch_size,1, hidden_dim)
            # 각각을 projection 후 더하기 -> tanh
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))  #
            return out.matmul(self.va).squeeze(
                -1)  # (batch_size,max_timestep,hidden_dim) ->  (batch_size, max_timestep)

        else:
            raise NotImplementedError


class Seq2Seq(nn.Module):
    """
        Sequence to sequence module
    """

    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.SOS = config.get("start_index", 1)  # Start index를 가져옵니다.
        self.vocab_size = config.get("n_classes", 32)  # embedding 에 필요한 vocabulary size
        self.batch_size = config.get("batch_size", 1)  # batch_size 정보를 가져옵니다.
        self.gpu = config.get("gpu", False)  # cuda 로 돌아가는지 아닌지에 대한 정보

        # Encoder 선언

        self.encoder = EncoderRNN(config)

        # Decoder 선언

        self.decoder = BahdanauDecoder(config)

        # loss fucntion
        # ignore_index =0 왜???
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    def encode(self, x, x_len):
        # encoder를 통해 주어진 source 정보를 Encodeing 하는 용도

        batch_size = x.size()[0]
        # 초기 inital hidden state 만들기
        init_state = self.encoder.init_hidden(batch_size)
        # encoder Forward 수행
        encoder_outputs, encoder_state = self.encoder.forward(x, init_state, x_len)

        return encoder_outputs, encoder_state.squeeze(0)

    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths, input_lengths):
        """
        Args:
            encoder_outputs: (B, T, H)
            encoder_hidden: (B, H)
            targets: (B, L)
            targets_lengths: (B)
            input_lengths: (B)
        Vars:
            decoder_input: (B)
            decoder_context: (B, H)
            hidden_state: (B, H)
            attention_weights: (B, T)
        Outputs:
            alignments: (L, T, B)
            logits: (B*L, V)
            labels: (B*L)
        """

        batch_size = encoder_outputs.size()[0]
        max_length = targets.size()[1]
        # decoder의 처음 y0 는 무엇이 되어야 할까? *주의해야할 포인트
        if batch_size == 1:
            decoder_input = torch.LongTensor([self.SOS] * batch_size)
        else:
            decoder_input = torch.LongTensor([self.SOS] * batch_size).squeeze(-1)
        decoder_context = encoder_outputs.transpose(1, 0)[-1]  # (Batch,H)
        decoder_hidden = encoder_hidden

        # alignments :  attention align을 저장하기 위한 용도
        alignments = torch.zeros(max_length, encoder_outputs.size(1),
                                 batch_size)  # attention align을 저장하기 위한 용도
        logits = torch.zeros(max_length, batch_size,
                             self.decoder.output_size)  # logits 값을 저장하기 위한 용도의 tensor

        if self.gpu:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            logits = logits.cuda()
        inference = []
        for t in range(max_length):

            # The decoder accepts, at each time step t :
            # - an input, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - encoder outputs, [B, T, H]

            # The decoder outputs, at each time step t :
            # - an output, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - weights, [B, T]

            outputs, decoder_hidden, attention_weights = self.decoder.forward(
                input=decoder_input.long(),
                prev_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                seq_len=input_lengths)

            alignments[t] = attention_weights.transpose(1, 0)

            logits[t] = outputs

            if self.training:
                decoder_input = targets[:, t]
            else:
                topv, topi = outputs.data.topk(1)  # 가장 높은 예측만 사용.
                decoder_input = topi.squeeze(-1).detach()
                inference.append(decoder_input.cpu())

        labels = targets.contiguous().view(-1)

        mask_value = 0
        # what is this mask_3d? # (warning check)
        logits = mask_3d(logits.transpose(1, 0), targets_lengths, mask_value)
        logits = logits.contiguous().view(-1, self.vocab_size)  # loss를 구하기 위해 쫙 펴주기

        return logits, labels.long(), alignments, inference

    def step(self, batch):
        x, y, x_len, y_len = batch
        if self.gpu:
            x = x.cuda()
            y = y.cuda()
            x_len = x_len.cuda()
            y_len = y_len.cuda()

        encoder_out, encoder_state = self.encode(x, x_len)  # encoder
        logits, labels, alignments, inference = self.decode(encoder_out, encoder_state, y, y_len,
                                                            x_len)  # decoder 를 통해 alignment와 logit 값 얻기
        return logits, labels, alignments, inference

    def loss(self, batch):
        logits, labels, alignments, inference = self.step(batch)
        loss = self.loss_fn(logits, labels)  # loss 구하기 우리는 cross entropy 사용
        return loss, logits, labels, alignments, inference


def train(model, optimizer, train_loader, epoch, n_epochs):
    losses = []
    cers = []

    model.train()  # train mode
    count = 0
    for batch in train_loader:
        loss, _, _, _, _ = model.loss(batch)
        losses.append(loss.item())
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()

    print('\n [{}/{}] avg_loss= {:05.3f}'.format(epoch, n_epochs, np.mean(losses)))

    return model, optimizer


def evaluate(model, eval_loader):
    losses = []
    accs = []
    edits = []

    model.eval()  # why??

    with torch.no_grad():
        for batch in eval_loader:
            # t.set_description(" Evaluating... (train={})".format(model.training))
            loss, logits, labels, alignments, _ = model.loss(batch)
            preds = logits.detach().cpu().numpy()

            acc = 100 * np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            edit = editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(
                preds)

            losses.append(loss.item())

            accs.append(acc)
            edits.append(edit)

        align = alignments.detach().cpu().numpy()[:, :, 0]

    print("  End of evaluation : loss {:05.3f} , acc {:03.1f} , edits {:03.3f}".format(
        np.mean(losses), np.mean(accs), np.mean(edits)))

# 시각화
def draw(data, x, y):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                    cbar=False)
    plt.show()


def visualize_plot(model, custom_input='cgdafa'):
    c_xs = np.array([dataset.char2int.get(x) for x in custom_input] + [2])
    c_xs = torch.from_numpy(c_xs).unsqueeze(0).long()

    c_xl = torch.tensor(c_xs[0].size()[-1]).unsqueeze(0)

    c_ys = np.array([dataset.char2int.get(x) for x in custom_input[::-1]] + [
        2])  # Return the random string and its reverse + EOS
    c_ys = torch.from_numpy(c_ys).unsqueeze(0).long()

    c_yl = torch.tensor(c_ys[0].size()[-1]).unsqueeze(0)
    c_data = (c_xs, c_ys, c_xl, c_yl)
    loss, logits, labels, alignments, predict = model.loss(c_data)
    heat_map_value = alignments.detach().cpu().numpy()[:, :, 0]
    preds = logits.detach().cpu().numpy()
    preds = np.argmax(preds, -1)
    source_tokens = [dataset.int2char[item - 3] for item in c_xs[0] if item != 0 if item != 2] + [
        '</s>']
    target_tokens = [dataset.int2char[item - 3] if item != 2 else '</s>' for item in preds.tolist()
                     if item != 0]
    draw(heat_map_value, source_tokens, target_tokens)

def main():
    USE_CUDA = torch.cuda.is_available()
    batch_size = 32
    epochs = 6
    dataset = ToyDataset(5, 15)
    eval_dataset = ToyDataset(5, 15, type='eval')

    train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                   collate_fn=pad_collate, drop_last=True)
    eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=pad_collate, drop_last=True)

    config = {
        "decoder": "Bahdanau",
        "encoder": "RNN",
        "n_channels": 4,
        "encoder_hidden": 64,
        "encoder_layers": 1,
        "encoder_dropout": 0.2,
        "bidirectional_encoder": True,
        "decoder_hidden": 64,
        "decoder_layers": 1,
        "decoder_dropout": 0.2,
        "n_classes": dataset.VOCAB_SIZE + 3,
        "batch_size": 32,
        "embedding_dim": 64,
        "attention_score": "bahdanau",
        "learning_rate": 0.001,
        "gpu": True,
        "loss": "cross_entropy"
    }

    model = Seq2Seq(config)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    for epoch in range(epochs):
        model, optimizer = train(model, optimizer, train_loader, epoch, epochs)
        evaluate(model, eval_loader)


    visualize_plot(model, 'cbada')

    config = {
        "decoder": "Bahdanau",
        "encoder": "RNN",
        "n_channels": 4,
        "encoder_hidden": 64,
        "encoder_layers": 1,
        "encoder_dropout": 0.2,
        "bidirectional_encoder": False,
        "decoder_hidden": 64,
        "decoder_layers": 1,
        "decoder_dropout": 0.2,
        "n_classes": dataset.VOCAB_SIZE + 3,
        "batch_size": 32,
        "embedding_dim": 64,
        "attention_score": "concat",
        "learning_rate": 0.001,
        "gpu": True,
        "loss": "cross_entropy"
    }

    model = Seq2Seq(config)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    for epoch in range(epochs):
        model, optimizer = train(model, optimizer, train_loader, epoch, epochs)
        evaluate(model, eval_loader)

    visualize_plot(model, 'cbada')

    config = {
        "decoder": "Bahdanau",
        "encoder": "RNN",
        "n_channels": 4,
        "encoder_hidden": 64,
        "encoder_layers": 1,
        "encoder_dropout": 0.2,
        "bidirectional_encoder": False,
        "decoder_hidden": 64,
        "decoder_layers": 1,
        "decoder_dropout": 0.2,
        "n_classes": dataset.VOCAB_SIZE + 3,
        "batch_size": 32,
        "embedding_dim": 64,
        "attention_score": "concat",
        "learning_rate": 0.001,
        "gpu": True,
        "loss": "cross_entropy"
    }

    model = Seq2Seq(config)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    for epoch in range(epochs):
        model, optimizer = train(model, optimizer, train_loader, epoch, epochs)
        evaluate(model, eval_loader)

    visualize_plot(model, 'cbada')


if __name__ == "__main__":
    main()

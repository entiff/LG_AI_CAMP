import torch
from dataset import Language, NmtDataset
from model import Seq2Seq
from run import plot_attention

attention_type = 'concat'  # 'dot' or 'concat'
embedding_dim = 128
hidden_dim = 64
bucketing = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    french_train = Language(path='data/train.fr.txt')
    english_train = Language(path='data/train.en.txt')
    french_train.build_vocab()
    english_train.build_vocab()
    model = Seq2Seq(french_train, english_train, attention_type=attention_type,
                    embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load("seq2seq_" + attention_type + ".pth", map_location=device))

    french_test = Language(path='data/test.fr.txt')
    english_test = Language(path='data/test.en.txt')
    french_test.set_vocab(french_train.word2idx, french_train.idx2word)
    english_test.set_vocab(english_train.word2idx, english_train.idx2word)
    dataset = NmtDataset(src=french_test, trg=english_test)

    samples = [dataset[0][0], dataset[1][0], dataset[2][0]]  # You may choose your own samples to plot

    for i, french in enumerate(samples):
        translated, attention = model.translate(torch.Tensor(french).to(dtype=torch.long, device=device))
        source_text = [french_train.idx2word[idx] for idx in french]
        translated_text = [english_train.idx2word[idx] for idx in translated]
        plot_attention(attention.cpu().detach(), translated_text, source_text, name=attention_type + '_' + str(i))

    f = open('translated.txt', mode='w', encoding='utf-8')
    f_bleu = open('pred.en.txt', mode='w', encoding='utf-8')
    for french, english in tqdm(dataset, desc='Translated'):
        translated, attention = model.translate(torch.Tensor(french).to(dtype=torch.long, device=device))
        source_text = [french_train.idx2word[idx] for idx in french]
        target_text = [english_train.idx2word[idx] for idx in english if idx != SOS and idx != EOS]
        translated_text = [english_train.idx2word[idx] for idx in translated if idx != EOS]

        f.write('French    : ' + ' '.join(source_text) + '\n')
        f.write('English   : ' + ' '.join(target_text) + '\n')
        f.write('Translated: ' + ' '.join(translated_text) + '\n\n')
        f_bleu.write(' '.join(translated_text) + '\n')
    f.close()
    f_bleu.close()

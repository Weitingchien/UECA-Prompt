import argparse
import os
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import time
import numpy as np
from tqdm import tqdm

from utils import print_data_info
from training_logger import log_predictions

"""setting agrparse"""
parser = argparse.ArgumentParser(description='Training')

"""model struct"""
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--window_size', type=int, default=2, help='size of the emotion cause pair window')
parser.add_argument('--feature_layer', type=int, default=3, help='number of layer iterations')
parser.add_argument('--log_file_name', type=str, default='log', help='name of log file')
parser.add_argument('--model_type', type=str, default='ISML', help='type of model')
"""training"""
parser.add_argument('--folds', type=int, default=10, help='number of cross-validation folds to run')
parser.add_argument('--training_iter', type=int, default=20, help='number of train iterator')
parser.add_argument('--scope', type=str, default='Ind_BiLSTM', help='scope')
parser.add_argument('--batch_size', type=int, default=8, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for bert')
parser.add_argument('--usegpu', type=bool, default=True, help='gpu')
"""other"""
parser.add_argument('--test_only', type=bool, default=False, help='no training')
parser.add_argument('--checkpoint', type=bool, default=False, help='load checkpoint')
parser.add_argument('--checkpointpath', type=str, default='checkpoint/ECPE/', help='path to load checkpoint')
parser.add_argument('--savecheckpoint', type=bool, default=False, help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_ECPE', help='path to save checkpoint')
parser.add_argument('--device', type=str, default='0', help='device id')
parser.add_argument('--dataset', type=str, default='data_combine_ECPE/', help='path for dataset')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
use_gpu = False

if opt.usegpu and torch.cuda.is_available():
    print('use gpu')
    use_gpu = True
    print(f'using gpu {use_gpu}')


def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


class MyDataset(Dataset):
    def __init__(self, input_file, test=False, tokenizer=None):
        print('load data_file: {}'.format(input_file))
        self.x_bert, self.y_bert, self.label, self.mask_label = [], [], [], []
        self.gt_emotion, self.gt_cause, self.gt_pair = [], [], []
        self.doc_id = []
        self.test = test
        self.n_cut = 0 # 記錄因長度被截斷的數量
        self.tokenizer = tokenizer
        cnt_over_limit = 0 # 記錄長度超過512的數量 
        inputFile = open(input_file, 'r')
        # 讀取每篇文檔
        while True:
            line = inputFile.readline()
            if line == '': # 讀到空行，表示此文檔結束
                break
            line = line.strip().split() # e.g., '212 10' -> ['212', '10']
            self.doc_id.append(line[0]) # 儲存文檔ID: '212'
            d_len = int(line[1]) # 儲存子句的數量
            pairs = eval('[' + inputFile.readline().strip() + ']') # 讀取情感原因配對資訊'(10, 9)'，將
            pos, cause = zip(*pairs) # pos: 情感子句的位置 cause: 原因子句的位置

            full_document = ""
            mask_full_document = ""
            mask_label_full_document = ""
            part_sentence = []
            cnt_emotion_gt = 0
            cnt_cause_gt = 0
            cnt_pair_gt = 0

            for _ in range(d_len):
                words = inputFile.readline().strip().split(',')[-1]
                part_sentence.append(words)
            # 計算Ground Truth 情感、原因、配對數量，用於評分
            cnt_emotion_gt = len(set(pos)) # set: 去除重複
            cnt_cause_gt = len(set(cause))
            cnt_pair_gt = len(set(pairs))
            self.gt_emotion.append(cnt_emotion_gt)
            self.gt_pair.append(cnt_pair_gt)
            self.gt_cause.append(cnt_cause_gt)
            for i in range(1, d_len + 1):
                full_document = full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_label_full_document = mask_label_full_document + ' [MASK] ' + part_sentence[i - 1]
                # ------------ 建立答案模板 ------------
                # # 如果當前子句是情感子句
                if i in pos:
                    full_document = full_document + '是 '
                    # 如果同時也是原因子句
                    if i in cause:
                        full_document = full_document + '是 '
                        full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 '
                # 如果當前子句不是情感子句
                else:
                    full_document = full_document + '非 '
                    # 如果他是原因子句
                    if i in cause:
                        full_document = full_document + '是 '
                        full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                    # 既不是情感子句也不是原因子句
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 '

                full_document = full_document + '[SEP]' # 答案模板的結尾
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK][SEP]" # 題目模板
                mask_label_full_document = mask_label_full_document + "[MASK] [MASK] [MASK][SEP]"
            if (self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape !=
                    self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape):
                print('length wrong')

            count_len = len(self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])
            if count_len > 512:
                print("Over limit length{} document{}".format(count_len, line[0]))
                cnt_over_limit += 1
            mask_full_document = \
                self.tokenizer.encode_plus(mask_full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            full_document = \
                self.tokenizer.encode_plus(full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            mask_label_full_document = \
                self.tokenizer.encode_plus(mask_label_full_document, return_tensors="pt", max_length=512,
                                           truncation=True,
                                           pad_to_max_length=True)['input_ids']
            # 製作標準答案
            labels = full_document.masked_fill(mask_full_document != 103, -100)
            print(f'labels: {labels}')
            mask_labels = full_document.masked_fill(mask_label_full_document != 103, -100)

            self.x_bert.append(np.array(mask_full_document[0]))
            self.y_bert.append(np.array(full_document[0]))
            self.label.append(np.array(labels[0]))
            self.mask_label.append(np.array(mask_labels[0]))
        self.x_bert, self.y_bert, self.label, self.mask_label = map(np.array, [self.x_bert, self.y_bert, self.label,
                                                                               self.mask_label])
        self.gt_emotion, self.gt_cause, self.gt_pair = map(np.array, [self.gt_emotion, self.gt_cause, self.gt_pair])
        for var in ['self.x_bert', 'self.y_bert', 'self.label', 'self.mask_label', 'self.gt_emotion', 'self.gt_cause',
                    'self.gt_pair']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('n_cut {}'.format(self.n_cut))
        print('load data done!\n')

        self.index = [i for i in range(len(self.x_bert))]
        print("num_for_over_limit{}".format(cnt_over_limit))

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.doc_id[index], self.x_bert[index], self.y_bert[index], self.label[index], self.mask_label[index],
                     self.gt_emotion[index], self.gt_cause[index], self.gt_pair[index]]
        return feed_list

    def __len__(self):
        return len(self.x_bert)


class prompt_bert(torch.nn.Module):
    def __init__(self, bert_path='./bert-base-chinese'):
        super(prompt_bert, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, x_bert, labels):
        output = self.bert(x_bert, labels=labels)
        loss, logits = output.loss, output.logits
        return loss, logits


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}'.format(
        opt.batch_size, opt.learning_rate))
    print('training_iter-{}\n'.format(opt.training_iter))


def crf_prompt(logits, labels, x_bert, gt_emotion, gt_cause, gt_pair):
    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                   8203, 8459, 8325, 8454, 8473, 8273]
    emo_gt = torch.sum(gt_emotion)
    emo_pre = 0
    emo_acc = 0
    cause_gt = torch.sum(gt_cause)
    cause_pre = 0
    cause_acc = 0
    pair_gt = torch.sum(gt_pair)
    pair_pre = 0
    pair_acc = 0
    for i in range(labels.shape[0]):
        count_mask = -1
        j = 0
        count_sentence = 0
        while j < 512:
            if x_bert[i][j] == 103:
                count_mask += 1
                count_mask = count_mask % 3
                if count_mask == 0:
                    if labels[i][j] == 3221:
                        if torch.argmax(logits[i][j]) == 3221:
                            emo_acc += 1
                    if torch.argmax(logits[i][j]) == 3221:
                        emo_pre += 1

                if count_mask == 1:
                    if torch.argmax(logits[i][j]) == 3221:
                        cause_pre += 1
                    if labels[i][j] == 3221:
                        if torch.argmax(logits[i][j]) == 3221:
                            cause_acc += 1

                if count_mask == 2:
                    count_sentence += 1
                    mask = torch.zeros([21128])
                    case = [label_index[k] for k in range(max(0, -opt.window_size + count_sentence - 1),
                                                          min(75, opt.window_size + count_sentence))]
                    case.append(3187)
                    for index in case:
                        mask[index] = 1
                    logits_ = torch.argmax(logits[i][j] * mask)

                    if logits_ in label_index:
                        pair_pre += 1
                    if labels[i][j] in label_index:
                        if logits_ == labels[i][j]:
                            pair_acc += 1

                j = j + 1
            else:
                j = j + 1
    p_emotion = emo_acc / (emo_pre + 1e-8)
    p_cause = cause_acc / (cause_pre + 1e-8)
    p_pair = pair_acc / (pair_pre + 1e-8)
    r_emotion = emo_acc / (emo_gt + 1e-8)
    r_cause = cause_acc / (cause_gt + 1e-8)
    r_pair = pair_acc / (pair_gt + 1e-8)
    f_emotion = 2 * p_emotion * r_emotion / (p_emotion + r_emotion + 1e-8)
    f_cause = 2 * p_cause * r_cause / (p_cause + r_cause + 1e-8)
    f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-8)
    print('emo_gt {}  cause_gt {}  pair_gt {}'.format(emo_gt, cause_gt, pair_gt))
    return p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair


def run():
    if opt.log_file_name:
        save_path = opt.save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # sys.stdout = open(save_path + '/' + opt.log_file_name, 'w')

    print_time()
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    output_log_file = "training_output_log.json"

    # train
    print_training_info()  # 输出训练的超参数信息

    max_result_emo_f, max_result_emo_p, max_result_emo_r = [], [], []
    max_result_pair_f, max_result_pair_p, max_result_pair_r = [], [], []
    max_result_cause_f, max_result_cause_p, max_result_cause_r = [], [], []
    for fold in range(1, opt.folds + 1):
        # model
        print('build model..')
        model = prompt_bert(bert_path)
        print('build model end...')
        if opt.checkpoint:
            model = torch.load(opt.checkpointpath + '/fold{}.pth'.format(fold),
                               map_location=torch.device('cpu'))
        if use_gpu:
            model = model.cuda()

        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)
        print('############# fold {} begin ###############'.format(fold))
        train = opt.dataset + train_file_name
        test = opt.dataset + test_file_name
        edict = {"train": train, "test": test}
        NLP_Dataset = {x: MyDataset(edict[x], test=(x == 'test'), tokenizer=tokenizer) for x in ['train', 'test']}
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)

        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause, max_p_pair,\
        max_r_pair, max_f1_pair = [-1.] * 9
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        if opt.test_only:
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            all_test_emotion_gt = torch.tensor([])
            all_test_cause_gt = torch.tensor([])
            all_test_pair_gt = torch.tensor([])
            model.eval()
            with torch.no_grad():
                progress_bar_eval = tqdm(testloader, desc=f"Fold {fold} Epoch {i+1} - Evaluating")
                for _, data in enumerate(progress_bar_eval):
                    doc_ids, x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                    loss, logits = model(x_bert, label)
                    logits = F.softmax(logits, dim=-1)
                    all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                    all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                    all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                    all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                    all_test_x_bert = torch.cat((all_test_x_bert, x_bert.cpu()), 0)
                    all_test_emotion_gt = torch.cat((all_test_emotion_gt, gt_emotion), 0)
                    all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)
                    all_test_pair_gt = torch.cat((all_test_pair_gt, gt_pair), 0)

                p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                    all_test_logits, all_test_label, all_test_x_bert, all_test_emotion_gt, all_test_cause_gt,
                    all_test_pair_gt)
                print(
                    "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                    " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        p_emotion,
                        r_emotion,
                        f_emotion,
                        p_cause,
                        r_cause,
                        f_cause,
                        p_pair,
                        r_pair,
                        f_pair))
                if f_emotion > max_f1_emotion:
                    max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                if f_cause > max_f1_cause:
                    max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                if f_pair > max_f1_pair:
                    max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair

                print(
                    "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                    " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                        max_p_pair, max_r_pair, max_f1_pair))

        else:
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                progress_bar = tqdm(trainloader, desc=f"Fold {fold} Epoch {i+1}/{opt.training_iter}")
                for index, data in enumerate(progress_bar):
                    with torch.autograd.set_detect_anomaly(True):
                        doc_ids, x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
                        # print_data_info(data, tokenizer) #印出樣本的詳細資訊
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                        loss, logits = model(x_bert, mask_label)
                        logits = F.softmax(logits, dim=-1)
                        
                        
                        # Log predictions for each sample in the batch
                        batch_size_actual = x_bert.size(0) # Get current batch size
                        for batch_idx in range(batch_size_actual):
                            doc_id_single = doc_ids[batch_idx] # Get the specific doc_id from the batch
                            
                            x_bert_single = x_bert[batch_idx] 
                            predicted_y_bert_single = logits[batch_idx].argmax(dim=-1) 
                            
                            label_single = label[batch_idx] 
                            mask_label_single = mask_label[batch_idx] 
                            
                            gt_emotion_single = gt_emotion[batch_idx]
                            gt_cause_single = gt_cause[batch_idx]
                            gt_pair_single = gt_pair[batch_idx]

                            log_predictions(
                                doc_id=doc_id_single,
                                x_bert_ids=x_bert_single.cpu(),
                                predicted_y_bert_ids=predicted_y_bert_single.cpu(),
                                label_ids=label_single.cpu(),
                                mask_label_ids=mask_label_single.cpu(),
                                gt_emotion=gt_emotion_single.cpu(),
                                gt_cause=gt_cause_single.cpu(),
                                gt_pair=gt_pair_single.cpu(),
                                tokenizer=tokenizer, 
                                output_filepath=output_log_file
                            )

                        optimizer.zero_grad()
                        if use_gpu:
                            loss = loss.cuda()
                        loss.backward()
                        optimizer.step()
                    
                        progress_bar.set_postfix(loss=loss.item())# 當前批次的損失值顯示在進度條末端

                        # print("loss: {:.4f}".format(loss))
                        if index % 20 == 0:
                            p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = \
                                crf_prompt(logits.cpu(), label.cpu(), x_bert.cpu(), gt_emotion, gt_cause, gt_pair)
                            print(
                                "iter: {} e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                                " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                                    index,
                                    p_emotion,
                                    r_emotion,
                                    f_emotion,
                                    p_cause,
                                    r_cause,
                                    f_cause,
                                    p_pair,
                                    r_pair,
                                    f_pair))
                all_test_logits = torch.tensor([])
                all_test_label = torch.tensor([])
                all_test_mask_label = torch.tensor([])
                all_test_y_bert = torch.tensor([])
                all_test_x_bert = torch.tensor([])
                all_test_emotion_gt = torch.tensor([])
                all_test_cause_gt = torch.tensor([])
                all_test_pair_gt = torch.tensor([])

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(testloader):
                        doc_ids, x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                        loss, logits = model(x_bert, label)
                        logits = F.softmax(logits, dim=-1)
                        all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                        all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                        all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                        all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                        all_test_x_bert = torch.cat((all_test_x_bert, x_bert.cpu()), 0)
                        all_test_emotion_gt = torch.cat((all_test_emotion_gt, gt_emotion), 0)
                        all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)
                        all_test_pair_gt = torch.cat((all_test_pair_gt, gt_pair), 0)

                    p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                        all_test_logits, all_test_label, all_test_x_bert, all_test_emotion_gt, all_test_cause_gt,
                        all_test_pair_gt)
                    print("iter{} test result:".format(i))
                    print(
                        "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f} pair_p: {:.4f}"
                        " pair_r: {:.4f} pair_f: {:.4f}".format(
                            p_emotion,
                            r_emotion,
                            f_emotion,
                            p_cause,
                            r_cause,
                            f_cause,
                            p_pair,
                            r_pair,
                            f_pair))
                    if f_emotion > max_f1_emotion:
                        max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                    if f_cause > max_f1_cause:
                        max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                    if f_pair > max_f1_pair:
                        max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair
                        if opt.savecheckpoint:
                            torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))

                    print("iter{} test result:".format(i))
                    print(
                        "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                        " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                            max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                            max_p_pair, max_r_pair, max_f1_pair))
        max_result_emo_f.append(max_f1_emotion)
        max_result_cause_f.append(max_f1_cause)
        max_result_pair_f.append(max_f1_pair)
        max_result_emo_p.append(max_p_emotion)
        max_result_cause_p.append(max_p_cause)
        max_result_pair_p.append(max_p_pair)
        max_result_emo_r.append(max_r_emotion)
        max_result_cause_r.append(max_r_cause)
        max_result_pair_r.append(max_r_pair)

    print("emotion")
    print(max_result_emo_f)
    print("average f {:.4f}".format(sum(max_result_emo_f) / len(max_result_emo_f)))
    print(max_result_emo_p)
    print("average p {:.4f}".format(sum(max_result_emo_p) / len(max_result_emo_p)))
    print(max_result_emo_r)
    print("average r {:.4f}".format(sum(max_result_emo_r) / len(max_result_emo_r)))
    print("cause")
    print(max_result_cause_f)
    print("average f {:.4f}".format(sum(max_result_cause_f) / len(max_result_cause_f)))
    print(max_result_cause_p)
    print("average p {:.4f}".format(sum(max_result_cause_p) / len(max_result_cause_p)))
    print(max_result_cause_r)
    print("average r {:.4f}".format(sum(max_result_cause_r) / len(max_result_cause_r)))
    print("pair")
    print(max_result_pair_f)
    print("average f {:.4f}".format(sum(max_result_pair_f) / len(max_result_pair_f)))
    print(max_result_pair_p)
    print("average p {:.4f}".format(sum(max_result_pair_p) / len(max_result_pair_p)))
    print(max_result_pair_r)
    print("average r {:.4f}".format(sum(max_result_pair_r) / len(max_result_pair_r)))


if __name__ == '__main__':
    run()

__author__ = 'Haohan Wang'

import numpy as np


def loadDiagnosis():
    text = [line.strip() for line in open('../data/ADNI/diagnosis/split.pretrained.0.csv')]
    subject = {}

    for line in text:
        items = line.split(',')
        sid = items[0][-4:]
        eid = items[1]
        diag = items[4]

        if diag == 'AD':
            subject[sid + '#' + eid] = 1
        elif diag == 'CN':
            subject[sid + '#' + eid] = 0

    return subject

def loadCovariate():
    text = [line.strip() for line in open('../data/ADNI/diagnosis/split.pretrained.0.csv')][1:]
    subject_age = {}
    subject_gender = {}

    for line in text:
        items = line.split(',')
        sid = items[0][-4:]
        eid = items[1]
        age = items[2]

        subject_age[sid + '#' + eid] = float(age)

        gender = items[3]
        if gender.startswith('F'):
            subject_gender[sid + '#' + eid] = 0
        else:
            subject_gender[sid + '#' + eid] = 1

    return subject_age, subject_gender


def loadEmbedding(embedding_folder):

    embd = {}

    text = [line.strip() for line in
            open('../data/ADNI/' + embedding_folder + '/result_aug_fold_0_seed_1_epoch_50_ADNI.csv')]
    embedding = np.load('../data/ADNI/' + embedding_folder + '/result_aug_fold_0_seed_1_epoch_50_ADNI.npy')
    for i in range(len(text)):
        line = text[i]

        items = line.split(',')
        sid = items[0][-4:]
        eid = items[1]
        embd[sid + '#' + eid] = embedding[i]

    text = [line.strip() for line in
            open('../data/ADNI/' + embedding_folder + '/result_aug_fold_0_seed_1_epoch_50_ADNI_train.csv')]
    embedding = np.load('../data/ADNI/' + embedding_folder + '/result_aug_fold_0_seed_1_epoch_50_ADNI_train.npy')
    for i in range(len(text)):
        line = text[i]

        items = line.split(',')
        sid = items[0][-4:]
        eid = items[1]
        embd[sid + '#' + eid] = embedding[i]

    text = [line.strip() for line in
            open('../data/ADNI/' + embedding_folder + '/result_aug_fold_0_seed_1_epoch_50_ADNI_val.csv')]
    embedding = np.load('../data/ADNI/' + embedding_folder + '/result_aug_fold_0_seed_1_epoch_50_ADNI_val.npy')
    for i in range(len(text)):
        line = text[i]

        items = line.split(',')
        sid = items[0][-4:]
        eid = items[1]
        embd[sid + '#' + eid] = embedding[i]

    return embd

def loadGeneExpression():
    text = [line.strip() for line in open('../data/ADNI/GeneExpression/samples.txt')]

    expression = np.load('../data/ADNI/GeneExpression/ge.npy')

    exp = {}

    for i in range(len(text)):
        line = text[i]
        sid = line[-4:]

        exp[sid] = expression[i]

    return exp

def loadSNP():
    text = [line.strip() for line in open('../data/ADNI/SNPs/samples.txt')]
    snps = np.load('../data/ADNI/SNPs/snps.npy').astype(float)

    s = {}

    for i in range(len(text)):
        line = text[i]
        sid = line[-4:]

        s[sid] = snps[i]

    return s

def onlySelectExon(snps):

    text1 = [line.strip() for line in open('../data/ADNI/SNPs/markers.txt')]
    text2 = [line.strip() for line in open('../data/ADNI/SNPs/exonMarkers.txt')]

    result = np.zeros([snps.shape[0], len(text2)])

    for i in range(len(text2)):
        idx = text1.index(text2[i])

        result[:, i] = snps[:, idx]

    return result



def prepareData(embedding_folder):

    label = loadDiagnosis()
    age, gender = loadCovariate()
    snps = loadSNP()

    embedding = loadEmbedding(embedding_folder)

    data_label = []
    data_snps = []
    data_embedding = []
    data_covariate = []

    ids = []

    discussed = {}

    for seid in label:
        if seid in embedding:
            sid, eid = seid.split('#')

            if sid not in discussed:

                discussed[sid] = 0 # this is to be discussed

                if sid in snps:
                    data_label.append(label[seid])
                    data_embedding.append(embedding[seid])
                    data_snps.append(snps[sid])
                    data_covariate.append([age[seid], gender[seid]])
                    ids.append(seid)

    data_label = np.array(data_label)
    data_snps = np.array(data_snps)
    data_embedding = np.array(data_embedding)
    data_covariate = np.array(data_covariate)

    data_snps = onlySelectExon(data_snps)

    print (data_label)
    print (np.sum(data_label))  # 320

    print (data_label.shape)
    print (data_snps.shape)
    print (data_embedding.shape)
    print (data_covariate.shape)

    np.save('../data/ADNI/prepared/labels.npy', data_label)
    np.save('../data/ADNI/prepared/snps.npy', data_snps)
    np.save('../data/ADNI/prepared/embeddings.npy', data_embedding)
    np.save('../data/ADNI/prepared/covariates.npy', data_covariate)
    f = open('../data/ADNI/prepared/ids.txt', 'w')
    for id in ids:
        f.writelines(id + '\n')
    f.close()

if __name__ == '__main__':

    embeddingFolder = 'embeddings_aug'

    prepareData(embeddingFolder)

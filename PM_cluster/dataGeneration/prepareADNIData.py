__author__ = 'Haohan Wang'

import numpy as np

def preprocessGeneExpressionData():
    text = [line.strip() for line in open('../data/ADNI/GeneExpression2/ADNI_Gene_Expression_Profile.csv')]

    gene_of_interest = [line.strip() for line in open('../data/ADNI/topGenes.txt')]

    sid = []
    geneNames = []
    genes_tr = []
    for i in range(len(text)):
        line = text[i]
        if i == 2:
            items = line.split(',')
            for k in range(3,len(items)):
                sid.append(items[k])

        if i > 9:
            items = line.split(',')
            geneName = items[2].split('||')[0].strip()

            if geneName in gene_of_interest:
                if geneName not in geneNames:
                    geneNames.append(geneName)
                    gene_tr = []
                    for k in range(3,len(items)-1):
                        gene_tr.append(float(items[k]))
                    genes_tr.append(gene_tr)


    genes_tr = np.array(genes_tr)
    gene = genes_tr.T

    print (gene.shape)
    print (len(set(sid)))
    print (len(set(geneNames)))

    np.save('../data/ADNI/GeneExpression/genes.npy', gene)

    f = open('../data/ADNI/GeneExpression/samples.txt', 'w')
    for s in sid:
        f.writelines(s + '\n')
    f.close()

    f = open('../data/ADNI/GeneExpression/geneNames.txt', 'w')
    for n in geneNames:
        f.writelines(n + '\n')
    f.close()


def load_and_save_Data():
    snps = np.load('../data/ADNI/SNPs/snps.npy')

    snp_sample = [line.strip().replace('_','') for line in open('../data/ADNI/SNPs/samples.txt')]

    gene = np.load('../data/ADNI/GeneExpression/genes.npy')

    print (gene.shape)

    gene_sample = [line.strip().replace('_','') for line in open('../data/ADNI/GeneExpression/samples.txt')]

    diag_text = [line.strip() for line in open('../data/ADNI/diagnosis/split.pretrained.0.csv')]

    diag_id = {}

    for line in diag_text:
        items = line.split(',')

        if items[4] == 'AD':
            diag_id[items[0][8:]] = 1
        elif items[4] == 'CN':
            diag_id[items[0][8:]] = 0

    snp_result = []
    ge_result = []
    diag_result = []
    sid_result = []

    for i in range(len(snp_sample)):
        sid = snp_sample[i]
        if sid in gene_sample:
            if sid in diag_id:
                sid_result.append(sid)
                diag_result.append(diag_id[sid])
                snp_result.append(snps[i,:])

                j = gene_sample.index(sid)

                ge_result.append(gene[j,:])

    print (len(diag_result))

    diag_result = np.array(diag_result)
    snp_result = np.array(snp_result)

    snp_result = selectExonMarkers(snp_result)

    ge_result = np.array(ge_result)


    print (diag_result)

    print (snp_result.shape)
    print (ge_result.shape)


    np.save('../data/ADNI/preparedForPersonalizedRegression/snps.npy', snp_result)
    np.save('../data/ADNI/preparedForPersonalizedRegression/ge.npy', ge_result)
    np.save('../data/ADNI/preparedForPersonalizedRegression/label.npy', diag_result)

    f = open('../data/ADNI/preparedForPersonalizedRegression/samples.txt', 'w')

    for sid in sid_result:
        f.writelines(sid+'\n')

    f.close()

def selectExonMarkers(snps):

    markers = [line.strip() for line in open('../data/ADNI/SNPs/markers.txt')]

    interestMarkers = [line.strip() for line in open('../data/ADNI/AlzMarker_clean.txt')]

    newSnp = np.zeros([snps.shape[0], len(interestMarkers)])

    c = 0

    for i in range(snps.shape[1]):
        m = markers[i]
        if m in interestMarkers:
            j = interestMarkers.index(m)
            newSnp[:,j] = snps[:,i]

            c += 1

    print (c)

    return newSnp



if __name__ == '__main__':
    preprocessGeneExpressionData()
    load_and_save_Data()


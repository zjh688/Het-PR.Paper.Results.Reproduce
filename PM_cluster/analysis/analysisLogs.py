__author__ = 'Haohan Wang'

import numpy as np

def readData():
    text = [line.strip() for line in open('logs.tsv')]

    result = {}

    for line in text:
        items = line.split('\t')
        key = '#'.join([str(i) for i in items[:4]])

        if key not in result:
            result[key] = [np.zeros((3, 10)), np.zeros(10)]

        result[key][0][int(items[8])-1][0] = float(items[10])
        result[key][0][int(items[8])-1][1] = float(items[12])
        result[key][0][int(items[8])-1][2] = float(items[18])
        result[key][0][int(items[8])-1][3] = float(items[24])
        result[key][0][int(items[8])-1][4] = float(items[11])
        result[key][0][int(items[8])-1][5] = float(items[17])
        result[key][0][int(items[8])-1][6] = float(items[23])
        result[key][0][int(items[8])-1][7] = float(items[14])
        result[key][0][int(items[8])-1][8] = float(items[20])
        result[key][0][int(items[8])-1][9] = float(items[26])

        print (result[key][0])

    for key in result:
        result[key][1] = np.std(result[key][0], 0)
        result[key][0] = np.mean(result[key][0], 0)

    return result

def writeDataOut(result):
    f = open('log_parsed.tsv', 'w')
    for k, v in sorted(list(result.items())):
        items = k.split('#')

        for m in items:
            f.writelines(m + '\t')
        for i in range(10):
            f.writelines(str(v[0][i]) + '\t')
            f.writelines(str(v[1][i]) + '\t')
        f.writelines('\n')

    f.close()

def organizedResults():
    result = readData()
    writeDataOut(result)


if __name__ == '__main__':
    organizedResults()

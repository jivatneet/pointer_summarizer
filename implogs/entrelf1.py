import sys,os,json
from fuzzywuzzy import fuzz
def extractpredentrels(line):
    predents = set()
    predrels = set()
    q = ''
    q = line.split('[sep]')[0]
    entsrels = line.split('[sep]')[1:]
    for phrase in entsrels:
        for word in phrase.split(' '):
            if word:
                if word[0] == 'p':
                    predrels.add(word)
                if word[0] == 'q':
                    predents.add(word)
    return q,predents,predrels
f = open(sys.argv[1])
lines = f.readlines()
tpe = 0
fpe = 0
fne = 0
tpr = 0
fpr = 0
fnr = 0
totalfuzz = 0
count = 0
exactmatch = 0

for idx,line in enumerate(lines):
    if 'target:' in line:
        print(line) #target
        print(lines[idx+1]) #answer
        
        #exact match
        target = lines[idx][9:].strip().split('[sep]')[0].split()
        ans = lines[idx+1][9:].strip().split('[sep]')[0].split()

        for ind,ele in enumerate(target):
            try:
              ele = float(ele)
              target[ind] = ele
            except ValueError:
                pass

        for ind, ele in enumerate(ans):
            try:
                ele = float(ele)
                ans[ind] = ele
            except ValueError:
                pass

        print("TARGET: ", target)
        print("ANS: ", ans)
        if target == ans:
            exactmatch += 1

        try:
            question,ents,rels = lines[idx].strip().split('[sep]')
        except Exception as err:
            print(err)
            continue
        goldents = set()
        goldrels = set()
        [goldents.add(x) for x in ents.split(' ') if x]
        [goldrels.add(x) for x in rels.split(' ') if x]
        #print(ents,rels,goldents,goldrels)
        predq,predents, predrels = extractpredentrels(lines[idx+1].strip())
        totalfuzz += fuzz.ratio(predq,question)
        '''
        if predq.split(' ')[2:-1] == question.split(' ')[2:-1]:
            print("exact match")
            exactmatch += 1
        '''
        count += 1
        print("goldents: ",goldents)
        print("goldrels: ",goldrels)
        print("predents: ",predents)
        print("predrels: ",predrels)
        #compute entity f1
        for goldentity in goldents:
            if goldentity in predents:
                tpe += 1
            else:
                fne += 1
        for queryentity in predents:
            if queryentity not in goldents:
                fpe += 1
        precisione = tpe/float(tpe+fpe+0.001)
        recalle = tpe/float(tpe+fne+0.001)
        f1e = 2*(precisione*recalle)/(precisione+recalle+0.001) 
        print("precisione: %f recalle: %f f1e: %f"%(precisione, recalle, f1e))
        #compute relation f1
        for goldrel in goldrels:
            if goldrel in predrels:
                tpr += 1
            else:
                fnr += 1
        for queryrel in predrels:
            if queryrel not in goldrels:
                fpr += 1
        precisionr = tpr/float(tpr+fpr+0.001)
        recallr = tpr/float(tpr+fnr+0.001)
        f1r = 2*(precisionr*recallr)/(precisionr+recallr+0.001)
        print("precisionr: %f recallr: %f f1r: %f"%(precisionr, recallr, f1r))
        print("question fuzz: ",float(totalfuzz)/count)
        print("exact query match %d out of %d"%(exactmatch,count))

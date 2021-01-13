import sys,os,json,copy,requests,rdflib

sparqloutlines = open(sys.argv[1]).readlines()
#lcqtestarr = json.loads(open('data/lcquad2sparqltestintermed1.json').read())
g = rdflib.Graph()
g.load('http://dbpedia.org/resource/Semantic_Web')
        
outdict = {}
right = 0
wrong = 0
count = 0
for idx,line in enumerate(sparqloutlines):
    if 'target:' in line:
        answer = sparqloutlines[idx+1][9:].strip()
        count += 1
        print(answer)
        answer = answer.replace('@@ent@@','<http://dbpedia.org/resource/International_Semantic_Web_Conference>').replace('@@pred@@','<http://dbpedia.org/ontology/academicDiscipline>').replace('[UNK]','\'12\'')
        # outdict[int(uid)] = answer
        print(answer)
        try:
            qres = g.query(answer)
            print(qres)
            for row in qres:
                print(row)
            right += 1
        except Exception as err:
            print(err)
            wrong += 1
        #url = 'https://query.wikidata.org/sparql'
        #query = answer
        #r = requests.get(url, params = {'format': 'json', 'query': query})
        #try:
        #    data = r.json()
        #    print(data)
        #    right += 1
        #except Exception as err:
        #    print("error:", err)
        #    wrong += 1
        print("count: ",count)
        print("right: ",right)
        print("wrong: ",wrong)
            

#finalarr = []
#for item in lcqtestarr:
#    ditem = copy.deepcopy(item)
#    #finalarr.append(ditem)
#    newd = {}
#    try:
#        newd['question'] = item['question'] + ' @@END@@ ' + outdict[item['uid']]
#        print("newdquestion: ", newd['question'])
#        newd['uid'] = item['uid']
#        newd['intermediate_sparql'] = item['intermediate_sparql']
#        finalarr.append(newd)
#    except Exception as err:
#        print(err)
#
#f = open('data/lcq2stagetestonly2.json','w')
#f.write(json.dumps(finalarr, indent=4, sort_keys=True))
#f.close()

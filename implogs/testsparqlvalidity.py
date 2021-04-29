import sys,os,json,copy,requests,rdflib

sparqloutlines = open(sys.argv[1]).readlines()
#lcqtestarr = json.loads(open('data/lcquad2sparqltestintermed1.json').read())
g = rdflib.Graph()
g.load('http://dbpedia.org/resource/Semantic_Web')
        
outdict = {}
right = 0
wrong = 0
count = 0
brace_mismatch = 0
exact_match = 0
lbrace = ['(', '{', '[']
rbrace = [')', '}', ']']
for idx,line in enumerate(sparqloutlines):
    l = 0
    r = 0
    if 'target:' in line:
        target = sparqloutlines[idx][9:].split('[sep]')[0]
        ans = sparqloutlines[idx+1][9:].split('[sep]')[0]
    
        if target.lower() == ans:
            exact_match += 1
        answer = ans.strip()
        
       #answer = sparqloutlines[idx+1][9:].strip()
        for ch in answer:
            if ch in lbrace:
                l += 1
            elif ch in rbrace:
                r += 1

        if l != r:
            brace_mismatch += 1
        count += 1
        #print(answer)
        answer = answer.replace('<entpos@@1>','<http://dbpedia.org/resource/International_Semantic_Web_Conference>').replace('<entpos@@2>','<http://dbpedia.org/resource/International_Semantic_Web_Conference>').replace('<entpos@@3>','<http://dbpedia.org/resource/International_Semantic_Web_Conference>').replace('<entpos@@4>','<http://dbpedia.org/resource/International_Semantic_Web_Conference>').replace('<entpos@@5>','<http://dbpedia.org/resource/International_Semantic_Web_Conference>').replace('<entpos@@6>','<http://dbpedia.org/resource/International_Semantic_Web_Conference>').replace('<predpos@@1>','<http://dbpedia.org/ontology/academicDiscipline>').replace('<predpos@@2>','<http://dbpedia.org/ontology/academicDiscipline>').replace('<predpos@@3>','<http://dbpedia.org/ontology/academicDiscipline>').replace('<predpos@@4>','<http://dbpedia.org/ontology/academicDiscipline>').replace('<predpos@@5>','<http://dbpedia.org/ontology/academicDiscipline>').replace('<predpos@@6>','<http://dbpedia.org/ontology/academicDiscipline>').replace('[UNK]','\'12\'')
        # outdict[int(uid)] = answer
        #print(answer)
        try:
            qres = g.query(answer)
            print(qres)
            for row in qres:
                print(row)
            right += 1
        except Exception as err:
            print("\nTARGET: ", target)
            print("QUERY: ", ans)
            print("WRONG SPARQL: ", answer)
            print(err)
            wrong += 1
            print("count: ", count)
            print("wrong: ", wrong)
            print("\n")
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
        #print("count: ",count)
        #print("right: ",right)
        #print("wrong: ",wrong)
        #print("brace mismatch: ", brace_mismatch)
        #print("exact match: ", exact_match)    

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

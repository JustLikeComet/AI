instances =[
"red", "green", "blue", "color"
]

relationship = ["is"]

facts = [
    ["red", "is", "color"], 
    ["blue", "is", "color"], 
    ["green", "is", "color"]
]


'''
ne / eq / gt / ge/ lt/ le
'''
logics = [
    {
        "name":"rule1", 
        "inputParams":["A", "B", "C", "D", "E"],
        "logics":[
#            ["r", "rule02", "arg01", "arg02"],  # for rule
            ["l", "A", "is", "color"], 
            ["l", "B", "is", "color"], 
            ["l", "C", "is", "color"], 
            ["l", "D", "is", "color"], 
            ["l", "E", "is", "color"],
            ["l", "A", "ne", "B"], 
            ["l", "A", "ne", "C"],
            ["l", "A", "ne", "D"],
            ["l", "A", "ne", "E"],
            ["l", "B", "ne", "C"],
            ["l", "C", "ne", "D"],
            ["l", "D", "ne", "E"]
        ]
    }
]

def initRuntimeVars(rule, tempRuntime, ruleVars):
    r = tempRuntime
    if r == None :
        r = {"vars":[], "logics":[], "result":[]}

    if ruleVars!=None:
        for var in ruleVars:
            if getVarFromRuntimeVar(r["vars"], var)==None:
                r["vars"].append( { "name":var, "currentValue":None, "currentIndex":-1 } )

    for i in range(len(rule["logics"])):
        r["logics"].append(False)
    return r

def getVarFromRuntimeVar(vars, name):
    for var in vars:
        if var["name"] == name:
            return var
    return None;

def getRuntimeVarWithVarsMap(inputParams, vars, name, varsmap):
    for i in range(len(inputParams)):
        if inputParams[i] == name:
            for var in vars:
                if varsmap[i] == var["name"]:
                    return var
    return None;

def getRule(rulename):
    for l in logics:
        if l["name"] == rulename:
            return l;
    return None

def checkLogics(index, ruleLogics, runtimeVars, varsMap):
    if index>=len(ruleLogics["logics"]):
        return True

    conditionValue = False

    # ne / eq / gt / ge/ lt/ le
    if ruleLogics["logics"][index][0] == 'l':
        vara = ruleLogics["logics"][index][1]
        varb = ruleLogics["logics"][index][3]
        if ruleLogics["logics"][index][2] != 'is':
            vara = getRuntimeVarWithVarsMap(ruleLogics["inputParams"], runtimeVars["vars"], vara, varsMap)["currentValue"]
            varb = getRuntimeVarWithVarsMap(ruleLogics["inputParams"], runtimeVars["vars"], varb, varsMap)["currentValue"]
        else:
            #print(vara)
            #print(runtimeVars["vars"])
            vara = getRuntimeVarWithVarsMap(ruleLogics["inputParams"], runtimeVars["vars"], vara, varsMap)["currentValue"]
        if ruleLogics["logics"][index][2] == 'ne':
            conditionValue = vara != varb
        elif ruleLogics["logics"][index][2] == 'eq':
            conditionValue = vara == varb
        elif ruleLogics["logics"][index][2] == 'gt':
            conditionValue = vara > varb
        elif ruleLogics["logics"][index][2] == 'ge':
            conditionValue = vara >= varb
        elif ruleLogics["logics"][index][2] == 'lt':
            conditionValue = vara < varb
        elif ruleLogics["logics"][index][2] == 'le':
            conditionValue = vara <= varb
        elif ruleLogics["logics"][index][2] == 'is':
            #print("relateship")
            #print("vara %s varb %s"%(vara, varb))
            conditionValue = checkRelateship(vara, 'is', varb)
        else:
            return False

    elif ruleLogics["logics"][index][0] == 'r':
        tempRule = getRule(ruleLogics["logics"][index][1])
        tempVarsMap = []
        for tempParam in ruleLogics["logics"][index][2:]:
            tempVarsMap.append(tempParam)
        conditionValue = checkRule(rule, tempRule, tempVarsMap)
    else:
        return False
    #print("%d %d %s %s %s %s %d"%(len(ruleLogics), index, ruleLogics[index][0],ruleLogics[index][1],ruleLogics[index][2],ruleLogics[index][3], conditionValue))
    if conditionValue :
        return checkLogics(index+1, ruleLogics, runtimeVars, varsMap)
    

def enumVarValue(index, vars):
    if index<0 or index>=len(vars):
        return False
    vars[index]["currentIndex"]+=1
    if vars[index]["currentIndex"]>=len(instances):
        return False
    #if vars[index]["currentIndex"]>=len(instances):
    #    vars[index]["currentIndex"] = 0
    vars[index]["currentValue"] = instances[ vars[index]["currentIndex"] ]
    #if vars[index]["currentIndex"]>=len(instances)-1:
    #    return False
    return True

def resetVarValue(index, vars):
    if index<0 or index>=len(vars):
        return 
    vars[index]["currentIndex"] = -1

def completeSearchAllWithRule(ruleLogics, runtimeVars, ruleVars ):
    mark = []
    while True:
        while len(mark)>=len(runtimeVars["vars"]):
            mark.pop()
        while len(mark)<len(runtimeVars["vars"]):
            hasNext = enumVarValue(len(mark), runtimeVars["vars"])
            if not hasNext and len(mark)==0:
                return
            if not hasNext:
                resetVarValue(len(mark), runtimeVars["vars"])
                mark.pop()
                continue
            mark.append(hasNext) 
        if checkRule(ruleLogics, runtimeVars, ruleVars):
            tempResult = []
            for var in runtimeVars["vars"]:
                tempResult.append({"name":var["name"], "value":var["currentValue"]})
            runtimeVars["result"].append(tempResult)


def checkRule(rule, runtimeVars, ruleVars):
    return checkLogics(0, rule, runtimeVars, ruleVars)

def listSrc(r, B):
    retList = []
    for fact in facts:
        if fact[1] == r and fact[2] == B:
            retList.append(fact[0])
    return retList

def listDest(A, r):
    retList = []
    for fact in facts:
        if fact[1] == r and fact[0] == A:
            retList.append(fact[2])
    return retList

def listRelateship(A, B):
    retList = []
    for fact in facts:
        if fact[2] == B and fact[0] == A:
            retList.append(fact[1])
    return retList

def checkRelateship(A, r, B):
    for fact in facts:
        if fact[2] == B and fact[0] == A and fact[1] == r:
            return True
    return False

print(listSrc("is", "color"))
print(listDest("blue", "is"))
print(listRelateship("blue", "color"))
tempRuntime = initRuntimeVars(logics[0], None, ["A", "B", "C", "D", "E"])
completeSearchAllWithRule(logics[0], tempRuntime, ["A", "B", "C", "D", "E"])
if len(tempRuntime["result"])>0:
    print(tempRuntime["result"])
    



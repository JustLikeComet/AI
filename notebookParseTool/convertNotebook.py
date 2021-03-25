import sys
import json
print(sys.argv[1])

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

if data!=None:
    print(data.keys())
    for cell in data["cells"]:
        if cell['cell_type']=='code':
            print("#########################################################")
            for srcline in cell['source']:
                print(srcline,end="")
            print("\n#########################################################")
    #print(data['metadata'])
    #print( data['nbformat'])
    #print(data[ 'nbformat_minor'])
    #for key in data.keys():
    #    print(data[key].keys())

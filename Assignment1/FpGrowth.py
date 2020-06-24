def id_to_vocab(file):
    vocabs = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            Id, vocab = line.split()
            vocabs[Id] = vocab
    return vocabs

def get_transactions(topic_file):
    transactions = []
    with open(topic_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            transactions.append(line.split())
    return transactions

class treenode:
    def __init__(self,value,number,parent):
        self.value=value       #node name
        self.number=number     #node count
        self.parent=parent     #parent node
        self.children={}       #chlidren node
        self.link=None         #node cross link

class confptreenode:
    def __init__(self,value,number,parent):
        self.value=value         #node name
        self.number=number       #node count
        self.parent=parent       #parent node
        self.children={}         #chlidren node

def updateconfptree(row,node,valuenum):
    if row[0] in node.children:
        node.children[row[0]].number=node.children[row[0]].number+valuenum
    else:
        node.children[row[0]]=treenode(row[0],valuenum,node)
    #recursive
    if len(row[1:])>=1:
        updateconfptree(row[1:],node.children[row[0]],valuenum)


def updatetree(row,node,headtable):
    if row[0] in node.children:
        node.children[row[0]].number=node.children[row[0]].number+1
    else:
        node.children[row[0]]=treenode(row[0],1,node)
        # headtabl enode link is empty, if there is no connection, create a new link
        if headtable[row[0]][1]==None:
            headtable[row[0]][1]=node.children[row[0]]
        else:
            tmp=headtable[row[0]][1]
            while(tmp.link!=None):
                tmp=tmp.link
            tmp.link=node.children[row[0]]
    #recursive
    if len(row[1:])>=1:
        updatetree(row[1:],node.children[row[0]],headtable)


def createtree(inputdataset, minsupprot):
    #create headtable frequent set
    dataset = inputdataset[:]
    headtable = {}
    for row in dataset:
        for item in row:
            if headtable.get(item) == None:
                headtable[item] = 1
            else:
                headtable[item] = headtable[item] + 1
    if '' in headtable.keys():
        del (headtable[''])
    for item in list(headtable.keys()):
        if headtable[item] < minsupprot:
            del (headtable[item])

    freqitemset = set(headtable.keys())
    #add link to tail
    for k in headtable:
        headtable[k] = [headtable[k], None]
    #filter
    for trans in dataset:
        for item in list(trans):
            if item not in freqitemset:
                trans.remove(item)
    while [] in dataset:
        dataset.remove([])
    # sort headtable by descending
    headtable = dict(sorted(headtable.items(), key=lambda x: x[1][0], reverse=True))
    rootnode = treenode('Null Set', 1, None)
    for row in dataset:
        sortrow = []
        for headtableitem in headtable:
            if headtableitem in row:
                sortrow.append(headtableitem)
        updatetree(sortrow, rootnode, headtable)
    return rootnode, headtable, freqitemset, dataset


def condfptreeonitem(headtable):
    allconfptree={}
    headtable=dict(sorted(headtable.items(), key = lambda x:x[1][0]))
    #find every element in the headtable
    for key,value in headtable.items():
        allconfptree[key]=[]
        tmp=value[1]
    #for each point on the link, look up parent until the rootnode
        while(tmp.link!=None):
            tmp2=tmp
            fptreebaseon={}
            tmpnumber=tmp2.number
            while(tmp2.value!='Null Set'):
                fptreebaseon[tmp2.value]=tmpnumber
                tmp2=tmp2.parent
            fptreebaseon[tmp2.value]=1
            allconfptree[key].append(fptreebaseon)
            tmp=tmp.link
    #find the parent of the end element in horizontal chain
        fptreebaseon={}
        tmpnumber=tmp.number
        while(tmp.value!='Null Set'):
            fptreebaseon[tmp.value]=tmpnumber
            tmp=tmp.parent
        fptreebaseon[tmp.value]=1
        allconfptree[key].append(fptreebaseon)
    return allconfptree

#find subset by binary
def subsetsbinary(datasets):
    N=len(datasets)
    subsets=[]
    for i in range(2**N):
        subset = []
        for j in range(N):
            if(i>>j)%2==1:
                subset.append(datasets[j])
        subsets.append(subset)
    del subsets[0]
    for i in list(subsets):
        if datasets[0] not in i:
            subsets.remove(i)
    return subsets


def countfptree(allconfptree,maxleafsize):
    allfrequentdataset={}
    allfrequentdataset1={}
    freqsetonkey=[]
    for key,confptreeonkey in allconfptree.items():
        tmp={}
        freqset={}
        #Calculate condition fptree about key
        for tree in confptreeonkey:
            for treeitem,num in tree.items():
                if treeitem in tmp:
                    tmp[treeitem]=tmp[treeitem]+num
                else:
                    tmp[treeitem]=num
                if 'Null Set' in tmp:
                    del tmp['Null Set']
        for tmpitem,num in tmp.items():
            if num>=maxleafsize:
                freqset[tmpitem]=num
        freqsetonkey.append(freqset)

        subsets=subsetsbinary(list(freqset.keys()))
        for subset in subsets:
            count=0
            lensubset=len(subset)
            for tree in confptreeonkey:
                itemcount=0
                for subsetitem in subset:
                    if subsetitem in tree:
                        itemcount=itemcount+1
                if itemcount==lensubset:
                    count=count+tree[subsetitem]
            if count>=maxleafsize:
                allfrequentdataset[tuple(subset)]=count
            allfrequentdataset1[tuple(subset)]=count
    return allfrequentdataset,freqsetonkey,allfrequentdataset1

def printconfptree(node):
    if len(node.children)!=0:
        print('[',end='')
    print('\"',end='')
    if node.value in id_dict:
        print(id_dict[node.value],end=' ')
    else:
        print(node.value, end=' ')
    print(node.number,end='')
    if len(node.children)!=0:
        print('\", ',end='')
        print('[',end='')
    if len(node.children)==1:
        printconfptree(node.children[list(node.children.keys())[0]])
        print(']',end='')
    if len(node.children)>1:
        i=0
        while(i<len(node.children)-1):
            printconfptree(node.children[list(node.children.keys())[i]])
            i=i+1
            print(',',end=' ')
        printconfptree(node.children[list(node.children.keys())[i]])
        print(']',end='')
    if len(node.children)!=0:
        print(']',end='')


def constructconfptree(freqsetonkey,allconfptree):
    heightfreqset={}
    for i in freqsetonkey:
        if len(i)>1:
            heightfreqset[list(i.keys())[0]]=list(i.keys())
    contree={}
    for key,values in allconfptree.items():
        if key in list(heightfreqset.keys()):
            contree[key]=confptreenode('Null Set',1,None)
            for row in values:
                newrow=[]
                for rowitem,rowvalue in row.items():
                    if rowitem in heightfreqset[key]:
                        newrow.append(rowitem)
                        count=rowvalue
                newrow.reverse()
                newrow.pop()
                if len(newrow)>0:
                    updateconfptree(newrow,contree[key],count)
    return contree


if __name__ == "__main__":

    id_dict = id_to_vocab('vocab.txt')

    input_file = 'topic-0.txt'  # change the file here
    output_file = input_file.replace('topic', 'pattern')

    trans = get_transactions(input_file)
    fptree, headtable, freqitemset, aftdataset = createtree(trans, 400)
    allconfptree = condfptreeonitem(headtable)
    frequentdataset, freqsetonkey, frequentdataset1 = countfptree(allconfptree, 400)
    # print(allfrequentdataset)

    with open(output_file, 'w') as f:
        freq_items = [(k, v) for k, v in frequentdataset.items()]
        freq_items.sort(key=lambda x: x[1], reverse=True)
        for fi in freq_items:
            freq_item = [id_dict[f] for f in fi[0]]
            f.write('{}\t{}\n'.format(fi[1], ' '.join(freq_item)))
    # Q2
    contree = constructconfptree(freqsetonkey, allconfptree)
    for key, value in contree.items():
        printconfptree(value)

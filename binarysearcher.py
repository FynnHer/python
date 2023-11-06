import random

liste = [1,2,3,4,12,13,14,23,24,25,49]
search = 50
begin = 0
end = len(liste)-1

def binarySearch (listBegin,listEnd,search):
    finished = False
    while finished == False:
        helper = listEnd - listBegin
        if helper == 1:
            print("Zahl nicht da")
            finished = True
            break
        leng = helper//2
        pos = listBegin + leng
        current = liste[pos]
        print(current)
        if current == search:
            print("Zahl gefunden an Stelle: ", pos)
            finished = True
        elif current < search:
            listBegin = pos
        elif current > search:
            listEnd = pos
        
binarySearch(begin,end,search)

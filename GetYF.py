import logging
from logging import handlers

if __name__ == '__main__':
    org=open('D:/datasets/chats/withnames.txt', 'r', encoding='UTF-8')
    onerole = open('yf.txt', 'w', encoding='UTF-8')
    list = org.readlines()
    writeflag=True
    for i in range(0, len(list)):
        if (list[i] == '杨帆\n' or list[i] == '帆\n' or list[i] == 'a F sha\n'):
            writeflag = True
            continue
        if (list[i] == '007RIN\n'):
            writeflag = False
            continue
        if(writeflag):
            onerole.write(list[i])
    org.close()
    onerole.close()



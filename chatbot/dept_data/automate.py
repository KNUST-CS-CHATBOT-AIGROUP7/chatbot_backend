import time

#run automation to run a python script every 20 seconds
while True:
    exec(open('c:/Users/Owusu-Turkson/Documents/GitHub/chatbot_backend/chatbot/dept_data/train.py').read())
    print(time.ctime())
    time.sleep(20)
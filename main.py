import telebot
import detector
import argparse
import datetime
import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from threading import Thread
from queue import Queue
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--video', '-v', type=str, default=0, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='dependencies/face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='dependencies/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
args = parser.parse_args()

token='5007823702:AAEplJU-WtGovsOwi-ivhv3G9Eh_u5wR02U'

detector10 = detector.Detector(
    face_detection_model=args.face_detection_model, 
    score_threshold=args.score_threshold,
    nms_threshold=args.nms_threshold,
    top_k=args.top_k,
    face_recognition_model=args.face_recognition_model,
    deviceId=args.video,
    scale=args.scale,
    place="10-ая аудитория"
)

src = "images/"

dirnames = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f))]

for directory in dirnames:
    path = src + directory
    onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    detector10.addUser(directory, onlyfiles)

bot=telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет! Это бот, который говорит кто был в последний раз замечан в лаборатории")

@bot.message_handler(commands=['check'])
def check_message(message):
    global detector10
    
    msg = "В последний раз были замечены:\n"
    lastSeen = detector10.lastSeen()
    for _, user in lastSeen.items():
        msg += "- {}: \n    Место: {} \n    Время: {}\n".format(user.name, user.place, user.time.strftime("%d %B %Y %H:%M:%S") if user.time is not None else user.time)
    
    bot.send_message(message.chat.id, msg)

@bot.message_handler(commands=['get_photo'])
def get_photo_handler(message):
    msg = "Фото какого пользователя?"
    lastSeen = detector10.lastSeen()
    for key, user in lastSeen.items():
        msg += "\n    {} - {}".format(key, user.name)
    mesg = bot.send_message(message.chat.id, msg)
    bot.register_next_step_handler(mesg, get_photo)


def get_photo(message):
    try:
        key = int(message.text)
    except Exception:
        bot.send_message(message.chat.id, "Не могу превратить твой ответ в номер")
        return
    
    lastSeen = detector10.lastSeen()
    if not key in lastSeen:
        bot.send_message(message.chat.id, "Нет пользователя с таким номером")
        return
        
    imagePathes = lastSeen[key].imagePathes
    if len(imagePathes) == 0:
        bot.send_message(message.chat.id, "У пользователя нет фото")
        return
    
    img = open(imagePathes[0], 'rb')
    bot.send_photo(message.chat.id, img)
    
@bot.message_handler(commands=['rename_user'])
def rename_user_handler(message):
    if message.from_user.username != "msc_15":
        bot.send_message(message.chat.id, "У тебя нет прав для этого. Если это ошибка, обратись к @msc_15")
        return
    
    msg = "Какого пользователя хочешь переименовать?"
    lastSeen = detector10.lastSeen()
    for key, user in lastSeen.items():
        msg += "\n    {} - {}".format(key, user.name)
        
    mesg = bot.send_message(message.chat.id, msg)
    bot.register_next_step_handler(mesg, rename_user_middle)


def rename_user_middle(message):
    try:
        key = int(message.text)
    except Exception:
        bot.send_message(message.chat.id, "Не могу превратить твой ответ в номер")
        return
    
    lastSeen = detector10.lastSeen()
    if not key in lastSeen:
        bot.send_message(message.chat.id, "Нет пользователя с этим номером")
        return
    
    imagePathes = lastSeen[key].imagePathes
    if len(imagePathes) == 0:
        bot.send_message(message.chat.id, "У пользователя нет фото")
        return
    
    img = open(imagePathes[0], 'rb')
    mesg = bot.send_photo(message.chat.id, img, caption="Введи новое имя.\nУчти, что если ввести существующее имя пользователя, то вся информация старого пользователя перейдёт к новому")
        
    bot.register_next_step_handler(mesg, lambda message: rename_user_final(message, key))

def rename_user_final(message, key):
    status = detector10.changeUser(key, message.text)
    msg = ""
    if status == 0:
        msg = "Нет пользователя с таким ID"
    elif status == 1:
        msg = "Имя не изменилось"
    elif status == 2:
        msg = "Успешно изменилось имя"
    else:
        msg = "Пользователь {} присоединился к пользователю {}".format(status[0], status[1])
    bot.send_message(message.chat.id, "Получилось! Новое имя " + message.text + " Ключ " + str(key))
    
    
def startTelegram():
    print("Start telegram")
    sys.stdout.flush()
    bot.polling()
    
def startDetector():
    global detector10
    print("Start Detector", flush=True)
    sys.stdout.flush()
    detector10.detection()



thread1 = Thread( target=startDetector )
thread2 = Thread( target=startTelegram )
thread1.daemon = True
thread2.daemon = True

thread1.start()
thread2.start()
# thread1.join()
# thread2.join()
while True: time.sleep(100)

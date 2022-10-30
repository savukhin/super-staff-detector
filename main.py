import telebot
import detector
import argparse

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

detector10.addUser("Сава", "images/Sava.png")

bot=telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет! Это бот, который говорит кто был в последний раз замечан в лаборатории")

@bot.message_handler(commands=['check'])
def check_message(message):
    msg = "В последний раз в 10-ой были замечены:\n"
    lastSeen = detector10.lastSeen()
    print(lastSeen)
    for _, user in lastSeen.items():
        print("USER IS", user)
        msg += "-{}: \n\tместо {} \n\tвремя {}".format(user.name, user.place, user.time)
    
    print(msg)
    bot.send_message(message.chat.id, msg)

print("Bot starting...")
bot.polling()
print("Bot started")

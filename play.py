from keras.models import load_model
import cv2
import numpy as np
from random import choice 

#cac truong hop
CASES  = ["rock", "paper", "scissors", "none"]

#cac thanh phan tao khung
X = [100, 500, 800, 1200, 50, 650, 400]
Y = [100, 500, 50, 600]
color = [(255,255,255), (0, 0, 255)] #BGR
thickness = [2, 4]
fontScale =[1,2]

#vung anh


def Moves(val):
    return CASES[val]


def calcWinner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

#load model train
model = load_model("rock-paper-scissors-model.h5")
#chụp video
cap = cv2.VideoCapture(0)

defaultMove = "none"
compMoveName = "none"
winner = "waiting..."
#thiết lập khung hình
cap.set(3, 1280) #width
cap.set(4, 720) #height

while cap.isOpened():
    #show video capture
    ret, frame = cap.read()
    
# =============================================================================
#     #kiểm tra nếu đọc khung hình ko đúng thì bỏ qua
#     if not ret:
#         continue 
# =============================================================================
   
    # tạo khung người chơi
    cv2.rectangle(frame, (X[0],Y[1]), (X[1],Y[0]), color[0], thickness[0])
    # tạo khung máy chơi
    cv2.rectangle(frame, (X[2],Y[1]), (X[3],Y[0]), color[0], thickness[0])
    
    #lấy hình ảnh nước đi của người chơi
    player = frame[Y[0]:Y[1], X[0]:X[1]]
    #tao khung mau cho video, chuyen bgr thanh rgb
    img = cv2.cvtColor(player, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # dự đoán input
    pred = model.predict(np.array([img]))
    moveIndex = np.argmax(pred[0])
    playerMoveName = Moves(moveIndex)

    # dự đoán thắng thua
    if defaultMove != playerMoveName:
        if playerMoveName != "none":
            compMoveName = choice(['rock', 'paper', 'scissors'])
            winner = calcWinner(playerMoveName, compMoveName)
        else:
            compMoveName = "none"
            winner = "Waiting..."
    defaultMove = playerMoveName

    # viết chú thích trên hình ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + playerMoveName, (X[4], Y[2]), font, fontScale[0], color[0], thickness[0], cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + compMoveName, (X[5], Y[2]), font, fontScale[0], color[0], thickness[0], cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner, (X[6], Y[3]), font, fontScale[1], color[1], thickness[1], cv2.LINE_AA)

    if compMoveName != "none":
        icon = cv2.imread("images/{}.png".format(compMoveName))
        icon = cv2.resize(icon, (400, 400))
        frame[Y[0]:Y[1], X[2]:X[3]] = icon
        
    cv2.imshow("Game keo bua bao", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

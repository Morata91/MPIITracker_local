import sys
import os

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import cv2
import dlib
import torch


from models.alexnet_based.MPIITrackerModel import MPIITrackerModel

# 画面の解像度を取得する
screen_width = 2560
# 上部メニューバーのサイズを考慮(高さ62ピクセル)
screen_height = 1664-62

#パス
CP_PATH = 'checkpoints/alexnet_based/fold00_best.pth.tar'

def main():
    # カメラをキャプチャする
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("カメラが開けませんでした")
        exit()

    # ウィンドウを名前付きで作成する
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    # フルスクリーンモードに設定する
    cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    # 赤い点の位置
    red_dot_position = (screen_width // 2, screen_height // 2)  # 例として画面の中央に設定
    
    detector = dlib.get_frontal_face_detector()


    while True:
        # フレームをキャプチャする
        ret, frame = cap.read()

        if not ret:
            print("フレームをキャプチャできませんでした")
            break

        # フレームを画面サイズにリサイズする
        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        
        face_detect(detector, frame_resized)
        
        # フレームに赤い点を描画する
        # 第1引数：描画対象の画像（frame_resized）第2引数：円の中心座標（red_dot_position）第3引数：円の半径（例では10ピクセル）第4引数：円の色（BGR形式で指定、例では赤色 (0, 0, 255)）第5引数：円の塗りつぶし（-1で塗りつぶし）
        cv2.circle(frame_resized, red_dot_position, 10, (0, 0, 255), -1)

        # フレームを表示する
        cv2.imshow('Camera', frame_resized)

        # 'q'キーが押されたらループを終了する
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()


def predict():
    x_max=150
    y_max=200
    binwidth_x = 30
    binwidth_y = 20
    
    xbins_num = x_max*2//binwidth_x
    ybins_num = y_max//binwidth_y
    
    model = MPIITrackerModel(xbins_num, ybins_num)
    
    saved = torch.load(CP_PATH)
    saved_state_dict = saved['state_dict']
    model.load_state_dict(saved_state_dict)
    model.eval()
    
    # with torch.no_grad():
    #     x, y, _ = model(imFace, imEyeL, imEyeR, faceGrid, eyeGrid)
    
def face_detect(detector, frame_resized):
    # dlibの顔検出器をロード
    # グレースケールに変換
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    # 顔検出
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # 顔のバウンディングボックスを描画
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)




if __name__ == '__main__':
    main()

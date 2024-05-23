import cv2

# 画面の解像度を取得する
screen_width = 2560
# 上部メニューバーのサイズを考慮(高さ62ピクセル)
screen_height = 1664-62

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


    while True:
        # フレームをキャプチャする
        ret, frame = cap.read()

        if not ret:
            print("フレームをキャプチャできませんでした")
            break

        # フレームを画面サイズにリサイズする
        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        
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


if __name__ == '__main__':
    main()

# AI-CUP-Competition
# 無人機飛行載具之智慧計數競賽
https://tbrain.trendmicro.com.tw/Competitions/Details/25

近年來科技高度發展所帶來之影響，已逐漸深入人類生活環境。無人飛行載具提供更寬廣的視野及更高度的移動性和靈活性，至今已被應用在許多不同產業領域，如地理資訊蒐集、交通監控、物品運送、通訊網路中繼站等不同類型。本次計畫著重於AI與影像辨識及無人飛行載具應用，期望參賽者以日常生活可能遭遇之問題為出發點，將深度學習原理作為基礎，運用相關人工智慧核心知識深入發展應用，進而將無人飛行載具相關技術應用於實際環境中，結合不同領域之人才並發揮創意，將平時所學專業知識確實落地，提升個人及團隊之實作能力與競爭力。 本次競賽主題為AI與影像辨識-無人機飛行載具之智慧計數（車輛與人群計數），無人機載具有高度移動性以及遠距遙控功能，能夠快速且輕易到達不容易接近的區域，搭配高解析度相機即如同鷹眼般，能從空中俯視地表，並將地表一切變化詳實記錄在影像中而不遺漏，目前國內尚無此空拍影像分析之比賽，此計畫將以無人機空拍影像為基礎，運用深度學習原理等相關訓練模組進行車輛與人群數量辨識。


競賽時間：2022/9/22 – 2022/12/5

參賽隊伍數：236隊

參賽隊伍名稱：TEAM_2110

Private Leaderboard : 48/236名

## 模型建置與訓練測試流程：  
### 第一版：  
首先根據每張圖片的長(height)寬(width)對其做標準化動作，因為 YOLOv5 輸入格式為標準化過後之數值。  
接著建置 YOLOv5x 模型設定參數：batch size, epoch, img size 等。  
![image](https://user-images.githubusercontent.com/110473288/209075418-8b3134ea-e3a5-4dc9-bdae-bacbf7515bb2.png)  
將資料集經過前處理後直接丟入YOLOv5x模型做Fine tune，並未作任何處理。  
Training score：
- Precision：0.782  
- Recall： 0.692  
- mAP50：0.739  
- mAP50-90：0.295  

Testing Hmean𝑇𝐼𝑜𝑈：0.597545  

### 第二版：  
重新分析資料集和bounding box，發現大部分的偵測目標都非常的小，儘管使用比Yolov5還要深度的Yolov5x（Testing Hmean𝑇𝐼𝑜𝑈 由0.585上升至0.597）但始終無法再有顯著的提升。  
因此對於Yolov5的Anchor先驗框，調整其原本預設的大小，將原本的Anchor值縮小，使得檢測目標的外框可以更加準確，依照Anchor先驗框再做調整優化，提升小目標偵測的精準度。  
![image](https://user-images.githubusercontent.com/110473288/209075559-b4d67542-7297-45a9-be3e-10d77c8a563d.png)  
Training score：
- Precision：0.85  
- Recall： 0.751  
- mAP50：0.817  
- mAP50-90：0.422  

Testing Hmean𝑇𝐼𝑜𝑈：0.649769 （0.597 -> 0.649）  

### 第三版（賽後持續改進）：
對於大部分的偵測小目標的任務，對Yolov5的架構做修改，降低初步CNN之layer的stride，使得模型Output的size變大，意味著每個feature的感受視野較小，特徵集中在小區域內，另外Anchor先驗框維持第二版的大小，以此來提升小目標偵測的精準度。  

修改：將backbone第一層CNN之stride設為1，使得Output採樣倍率降低。  

![image](https://user-images.githubusercontent.com/110473288/209767759-2b8c7c90-a466-43f8-816b-5e88909b53ce.png)  

Training score：（best）  
- Precision：0.881  
- Recall： 0.773  
- mAP50：0.842  
- mAP50-90：0.45  

## 最終結果：  
Score：0.642332  
最終名次：Top20%（48/236名）  

## 賽後結論：  
本次首次運用Yolo實作物件偵測，了解到物件偵測的原理與應用。往後能夠更加改進神經網路架構使得偵測更為準確、更符合data特性。  
在物件偵測上，主要的應用還是以Real Time較為重要，雖說本次使用較深的模型競賽，但往後會需要更加考慮快又準的輕巧模型。

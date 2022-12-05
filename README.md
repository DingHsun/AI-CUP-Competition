# AI-CUP-Competition
# 無人機飛行載具之智慧計數競賽（進行中） 
https://tbrain.trendmicro.com.tw/Competitions/Details/25

近年來科技高度發展所帶來之影響，已逐漸深入人類生活環境。無人飛行載具提供更寬廣的視野及更高度的移動性和靈活性，至今已被應用在許多不同產業領域，如地理資訊蒐集、交通監控、物品運送、通訊網路中繼站等不同類型。本次計畫著重於AI與影像辨識及無人飛行載具應用，期望參賽者以日常生活可能遭遇之問題為出發點，將深度學習原理作為基礎，運用相關人工智慧核心知識深入發展應用，進而將無人飛行載具相關技術應用於實際環境中，結合不同領域之人才並發揮創意，將平時所學專業知識確實落地，提升個人及團隊之實作能力與競爭力。 本次競賽主題為AI與影像辨識-無人機飛行載具之智慧計數（車輛與人群計數），無人機載具有高度移動性以及遠距遙控功能，能夠快速且輕易到達不容易接近的區域，搭配高解析度相機即如同鷹眼般，能從空中俯視地表，並將地表一切變化詳實記錄在影像中而不遺漏，目前國內尚無此空拍影像分析之比賽，此計畫將以無人機空拍影像為基礎，運用深度學習原理等相關訓練模組進行車輛與人群數量辨識。


競賽時間：2022/9/22 – 2022/12/5

目前參賽隊伍：236隊

參賽隊伍名稱：TEAM_2110

Public Leaderboard : 62/236名

模型建置與訓練測試流程：
首先根據每張圖片的長(height)寬(width)對其做標準化動作，因為 YOLOv5 輸入格式為標準化過後之數值。  
接著建置 YOLOv5x 模型設定參數：batch size, epoch, img size 等。  
並開始訓練120epoch觀測其結果。

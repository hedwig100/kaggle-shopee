# kaggle-shopee
kaggleのshopeeコンペのレポジトリ

![shopee_image](images/shopee_image.png) 

<br>

# 概要
- 同じ商品を見極める
- 与えられるデータは商品のタイトル(売った人が出した説明文的な)と商品の画像

<br>

# 指標
- 一つの商品ごとにその商品と同じ商品だと予測される他の商品を出力する
- 一つの商品ごとにF1scoreを計算してそれを平均する
- f1scoreは以下で表される量でrecallとprecisionのバランスをとっている. 

```
precision = TP/(TP + FP)
recall = TP/(TP + FN)
f1score = 2/(1/precision + 1/recall)
```

- f1scoreは閾値の設定も大事らしい(byKaggle本)

<br>

# 目標
- pytorchにもっと慣れる. 
- 2回目の画像コンペなので銅メダル
- googlecolabを有効活用する. 

<br>

# Notebook
[1. EDA](https://www.kaggle.com/ruchi798/shopee-eda-rapids-preprocessing) 
- edaのnotebook, 画像とかpandasの部分は自分でもできたが, NLPのとこはどんなことをすれば良いのかわからないので参考にする. 
- wordcloud
- bag-of-word : 単語に含まれる語彙をベクトルかしたもの. 
- 「stopword除去などの前処理 -> bag-of-wordでベクトルに変換 -> 類似している文をknnで探す.」 ということをした. 
- RAPIDSはGPUでの計算を効率よくするもの? [link](https://rapids.ai/)
- NLTKはNaturalLangageToolKitの略で自然言語処理に関するライブラリ. 

[2. Submission,CV](https://www.kaggle.com/underwearfitting/pytorch-densenet-arcface-validation-training)
- GroupKFoldでのCVをしている. 
- arcfaceという損失を用いたもの
- まずは普通にclass分類としてやってみてから, これをやってみる. 

<br>

# Log 

***20200324*** 
- コンペ参加し, 概要を把握した. 
- ざっとdiscussionとnotebookをみた. 自然言語処理っぽいところも重要らしい?
- arcfaceとか見慣れない単語がある. 
- 分類でも回帰でもないじゃん. それぞれのクラスについての分類として考えられなくはないけど, それはクラスごとにモデルを作ることになって現実的に不可能. 
- ↑こういう種類の問題ではmetric learningが使われるらしい. arcfaceとかはその時の指標の一つ? 
- ちょっと自分でedaをした. nb01
- みてた感じ似たような画像を使っている場合は画像で行けるかもしれないけど, titleの方が重要な情報を含んでいそう. 
- datasetをgoogle driveにuploadした. 

**20200325**
- nb01 
    - wordcloudを初めて使った. 英語ではない言語が入っている. 
    - edaのnotebookを参考にしつつbag-of-wordしてから, knnなどをした.
    - bag-of-word程度じゃ全然意味なさそう.
- とりあえずNLPパートは放っておいて, 画像パートをやる.  
- cvのやりかた
    - label-groupをつかってGroupKFold
    - train-setには含まれないがtest-setに含まれるlabelがあるかどうかが問題, [これに関するdiscusssion](https://www.kaggle.com/c/shopee-product-matching/discussion/224855). 
    - これはクラス分類(uniqueなclassは11014なので11014クラス)として捉えた時の話
    - とりあえずGroupKFoldでクラス分類の教師あり学習をしてみて, cvとlbの相関を見るべき. もし相関があるならtestdataもtraindataと同じlabelを持っている可能性が高いし, なければ全然違うlabelでできている可能性が高いと思う. 
    - nb02でとりあえずGroupKFoldした. 
- [このdiscussion](https://www.kaggle.com/c/shopee-product-matching/discussion/225543)を読んでいる限りだと, traindataと同じlabelを持っているとは考えられない? 
    - 普通にクラス分類としてtrainingしてから, softmaxをかける前の出力を用いてcosを求めるというのもありらしい. 
- nb04 
    - 普通のclass分類モデルを作り始めた. 

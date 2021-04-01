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

[3. CV](https://www.kaggle.com/tmhrkt/shopee-cv-splitting-way)
- CVをlabel_groupとimagehashを用いて分けている. imagehashやlabel_groupが同じものは同じfold内に存在しないように分けている. 
- image_phashにも気をつけないといけなかった. 

[4. unsupervised](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700) 
- textをtfidfでベクトル化して, KNNする
- いろんな言語が混じっているから, tfidfの方がBERTとかいうものよりも良いのかも? 

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
- nb03 
    - 普通のclass分類モデルを作り始めた. 
- transformer -> bert 
- 教師ありconstrastive learning

**20200329** 
- pytorchわかんないのでとりあえず, arcfaceでのtrainingをすることにした. 
- nb04 
    - とりあえず写経している.
    - 距離学習 [link](https://qiita.com/yu4u/items/078054dfb5592cbb80cc)
    - なんでarcfaceのとこでxを正規化してないの? 
    - と思ったらしてた. 
    - AMPはAutocast Mixed Precisionの略で計算の速度を早くする手法? 
    - デバッグしてとりあえず学習できた. 
    - 明日f1scoreの閾値最適化の部分と, submitcodeを書く. 
- [discussion](https://www.kaggle.com/c/shopee-product-matching/discussion/228537) 
    - 明日みよう
    - いろいろなこれまでの手法が見れる. 
- CVの切り方を変える. 
    - image_phashも考慮してGroupKFoldする. 
- OCR
    - 画像の中の文字をOCRで読み取って利用するという手もあるらしい

**20200330** 
- nb04 
    - thresholdoptimizerを実装した
    - 今のところ適当な範囲を指定して探索しているだけ
    - 学習がまだ終わってないけどlogをみている限り, バグっているような気がする. 
- colab
    - nb04をcolabでも学習できるように整備中
    - pytorchのメモリの管理ができない


**20200331**
- nb06
    - nb04の方法だと9時間で終わらないのでvalidationの頻度を減らして実行した
    - CV = 0.7384となった. -> LB 0.543
    - thresholdが0.6だと小さすぎるっぽい. ので0.8にしてみた. -> LB 0.658
    - LBに対してthresholdを最適化しないといけない...,PublicLBに対してoverfittingしちゃう...
    - 昨日のやり方だtrainlossを出力していなかったので, lossを出力するようにして, 再実行した. 
    - おそらくだけど, めちゃくちゃtraindataに過学習しているのでheavyaugmentationが効きそう? -> nb07
    - これは...
    ![overfitting](images/overfitting.png) 
    - 実はもっと小さいモデルeffcientnetB0とかの方がいい説がある. 
    - どんな画像に対して間違っているか確認する
- nb05
    - submit
    - 普通にsubmitするとメモリが足りないらしく, データを分けてcosin similarityを計算するようにした
- colabnb06
    - colabで実行できるようにした
- nb08 
    - どんな画像に対して間違っているか確認する
    ![mistak21](images/mistake1.png) 
    ![mistake2](images/mistake2.png) 

- textにも手を出したい
    - tfidf
    - bert
    - bertなら翻訳しないといけない? 

- [過去コンペのデータ](https://www.kaggle.com/terterter333/shopee-product-detection), 画像のみがあるらしい
    - 10GBくらい
    - クラスと画像があってこれで分類モデルをつくってから最後にこのコンペのデータでfinetuningする. 
    - データの被りが無いかどうかがきになる.


**20200401**
- nb05 
    - thresholdを0.75,0.85versionもやってみた. 
    |th|lb|
    |:--:|:--:|
    |0.65|0.546|
    |0.70|0.612|
    |0.75|0.646|
    |0.80|0.658|
    |0.85|0.658|
    - cvはth = 0.65のときに最大で0.7384だった. 
- nb07
    - めっちゃheavyなaugmentationしたら, 全然学習していなかった. 
    - まともなスコアが出ていないのでもうちょっと緩める
- nb09 
    - effcientnet_b3を試してみる. 
    - timmの使い方がよくわからん...
    - 画像サイズが小さいとbatchsizeが大きくできて嬉しい. 
- これからやるべきこと
    - weight regularizationありのdensenet121
    - channel_size(embeddingのサイズ)を小さくする
    - nfnetとかいうのを試す
    - tfidf
    - bert(翻訳が難しい?)

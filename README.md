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

[5. word2vec](https://www.kaggle.com/coder247/similarity-using-word2vec-text)
- wordをベクトルにembeddingsする方法
- simpleで使い方がわかりやすかった. 

[6. bert](https://www.kaggle.com/ragnar123/bert-baseline)
- tensorflowでbertの実装をしている. 
- やりたい. 

[7. sBert](https://www.kaggle.com/tanulsingh077/metric-learning-pipeline-only-text-sbert)
- sentence bertを使っている. 
- 多言語bert 
- pytorch 

[8. siamese net](https://www.kaggle.com/hedwig100/shopee-siamese-resnet-50-with-triplet-loss-on-tpu/edit)
- siamsese netはよくわからないが自分の理解だと, labelを用いずにtrainingしているので, labelのノイズの影響を受けない. 
- その意味で結構いいトレーニング方法かもしれない. 

[9. nfnet](https://www.kaggle.com/parthdhameliya77/shopee-pytorch-eca-nfnet-l0-image-training)
- 活性化関数をmishにして
- 最適化をRangerにするとよいらしい

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
    - cvはth = 0.65のときに最大で0.7384だった. 

|th|lb|
|:--:|:--:|
|0.65|0.546|
|0.70|0.612|
|0.75|0.646|
|0.80|0.658|
|0.85|0.658|

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

**20200402** 
- nb09 
    - effnet_b3のtrainingでもlossの下がり方から同じくらい過学習している気がする. 
    - **よく考えてみるとcos(\theta + m)とした後にlossを計算していてそれはなんか違くないかという感じになった.**
    - ↑これそうでもない気がしてきた. よくわかんない. 
    - cv = 0.715くらい(densenet121と同じエポック数で比べると少し良いcv)
    - weight_decayをするべき
    - weight_decay=1e-6でそれ以外の条件を全く変えないとcv = 0.7108だった. 
    - weight_decay強すぎ? もうちょっと小さくする? 
    - OptimizerがAdamWじゃないと意味ない説がある. 
    - colabで回す
- nb10
    - nb09のeffnetの提出をした. (weight_decayなし)
    - もしかしてcvとlbの相関が取れていそう? 
    - もうすこし考える必要あり
    - 下の表ははweight_decay=1e-6(cv = 0.7108)ver
    - やっぱり?相関取れてそう

|th|lb|
|:--:|:--:|
|0.70|0.540|
|0.80|0.634|


|th|lb|
|:--:|:--:|
|0.70|0.539|
|0.80|0.630|
|0.85|0.645|

- nb11
    - tfidfをする
    - 特に頑張ることがない
    - あえていうとstopwordsに特有のものを取り除くとscoreがあがったりするか? 
    - emojiとかは異常にたくさんあったし, それほど特徴として寄与しなさそうなので消したい
- todo
    - tfidfを回す
    - weight_decay=1e-7,1e-8くらいを回す
    - nfnet
    - bert 
    - channel_sizeを小さくする. 
    - 前のデータでpretrainする. 

**20200402**
- nb06 
    - densenet121でweight_decay=1e-4,AdamWで回したが, weight_decayなしの時より, val_f1scoreが減少した. (なし0.7384 -> あり0.7354)
    - weight_decayはSGDの方が効くらしい
    - やっぱりこのcvの切り方はめちゃくちゃ良さそう. 

|th|lb|
|:--:|:--:|
|0.70|0.610|
|0.80|0.658|
|0.90|0.647| 


- nb11
    - tfidfをした
    - th = 0.70で cv = 0.6139(ただtfidfをつくってcosine similarityを計算しただけなのでcvではないけど) 
    - todo
        - thを変える.
        - tfidfをnnに入れる

- nb13
    - label_smoothingつきcrossentropyを実装していたら, pytorchわかんないのでめちゃくちゃ時間かかった. 
    - colabが動いてくれない. 

- nb14 
    - textのembeddingとimageのembeddingをどうやってensembleするか考える
    - それぞれsimilarityを計算して
        - 足す?
        - 掛ける? 
        - max 
        - min 
    - それともちかいlabelを撮ってきてから
        - and 
        - or 

**20200404**
- nb13
    - label_smoothing=0.05のeffcientnetb3だとepoch=28,でcv = 0.75853だった. label_smoothingはかなり効いてるとおもう. 

|th|lb|
|:--:|:--:|
|0.80|0.650|
|0.85|0.660| 

- nb14
    - textのただtfidfしたやつと, effcientnetb3でcvを出したら, cv = 0.8118くらい
    - textの方はlabel_groupでgroup_kfoldしたのでこのCVがよいCvかどうかは不明. 
    - text_th = 0.70,image_th = 0.70
    - text_th = 0.80にすると, cv = 0.7941

- nb15
    - textとうえのeffb3で合わせてsubmitした. 
    - とりあえずCPUにしてしまった. 
    - 全く同じものをGPUでだした. 
    - GPUは15分くらい. 
    - text_th = 0.70だとlb = 0.571だったので少し調整すべき. 
    - text_th = 0.80だとlb = 0.621だった.
    - よく考えるとtestsizeの方がでかいからthももっと大きくすべきだなと持った. 
    - 正規化しているわけではないからcosでもないし, -> TfidfTransformerは内部で正規化している. 
    - submit時はexit()して時間を節約した.  

- nb16
    - nfnetを動かそうとしてみたが, あまりうまくいかなさそう. 
    - lossが発散しそう
    - というかもっと他にやるべき優先順位の高いことがある気がする. 

- todo
    - ranzcrで使われてた手法をためす. 
    - bert
    - channel_sizeを小さくする
    - 前のコンペのデータは使っていいか怪しいらしい. 

**20200405**
- nb15
    - 全く同じコードをgpuとcpuで提出したら結果が違ったので原因はgpuの倍精度演算の精度によるものと考えてすべてnp.float32にキャストしてから行列演算を行うものを提出してみた. 
    - gpuだと0.571でcpuだと0.707はさすがにおかしいとおもう. 
    - 確認してたらなぜかはよくわからないけれど, gpuではimageとtextのmatmulした後の計算結果が同じになっていた. 
    - なんで? 
    - ばかなミスがあった. 
    - 明日用の正しいsubを作っておいた. 

- nb01
    - 出現頻度の高く意味のなさそうな顔文字などの情報を消してみる. 
    - textの方をもっとちゃんと利用したい. 
    - dictみたいに与えられるデータに対して高速に文字列を置換する関数が欲しい. 
    - replaceがうまくいかない... 

- 過去コンペのデータは使えないらしい.

**20200406**
- nb06 
    - label_smoothingありだとcv = 0.7346817となった. 
    - なしより下がった

- nb15
    - いろいろミスがあったが, effnetのbestcvのモデルで以下のスコア
    - image単体モデルでのlb最高のsubmit th = 0.85で固定. 
    - image_cv = 0.7585
    - image_text_cv = 0.81180
    - (1fold分)
    - nb14(vesion8参照) 


|text_th|lb|
|:--:|:--:|
|0.70|0.707|
|0.80|0.703|
|0.90|0.687| 


    - densenetのbestcvのモデルでのスコア
    - image単体モデルでlb最高のsubmit th = 0.80で固定. 
    - image_cv = 0.7332888
    - image_text_cv = 0.800089
    - (1fold分) 
    - nb14(version11参照) 


|text_th|lb|
|:--:|:--:|
|0.70|0.714|
|0.80|0.707| 


    - あまりcv安定してない? 
    - textがunsupervised(tfidfしただけ)なのでそれと合わせると安定しなくなるのかも? 

- nb17
    - word2vecを理解する(must)
    - 理解してから実装をみる. 
    - [word2vec(Qiita)](https://qiita.com/Hironsan/items/11b388575a058dc8a46a)
    - word2vecは単語を入力として周辺にある単語が存在する確率を予測するモデルを作り, その重みを使うことで単語をベクトルにする. 
    - 訓練時の入力としては単語をonehot表現で表したもの. 
    - targetは周りにある単語とする.(これはwindowsizeにより変化する) 
    - 2層で隠れ葬の活性化関数に高等関数を用いているので, 結局近い単語のベクトルの内積を大きくするみたいな方向に訓練をする無用にしている.
    - 明日学習済みモデルとかをみる. 

**20200407** 
- nb13
    - effcientnetb3でもlabel_smoothingなしの方がcv = 0.75950で最高だった. 
    - やっぱりlabel_smoothingは意味ない説がある. 
    - text(tfidf)ありだと, cv = 0.8106122なのでtextと混ぜると前のlabel_smoothingありのeffnetの方がcvがよかったということになった. 

|th|lb|
|:--:|:--:|
|0.80|0.647|
|0.85|0.660| 
|0.90|0.657| 


    - 前のeffnetとlbはそんなに変わらない? 

- nb17
    - word2vecでtrainingしている. 
    - 学習済みモデルの方がいいんじゃないか感がある. 
    - gensimがlogを出してくれない. 
    - 公式のやり方でもできない. 
    - このデータしか使ってないからかわかんないけど, めちゃくちゃcvが悪い. cv = 0.12くらい. 

**20200410**
- nb13
    - effcientnetb3,label_smoothingなし, imsize=300でfold0では良かったのでfold1,fold2も同様に回した.
    - kaggleをやる時間が取れなかったのでとりあえず回しておく形にした. 

- nb17
    - word2vecは学習済みモデルを用いないで, 頑張るとcv = 0.511くらいが限界でtfidfの方がよかった. 
    - 英語の学習済みモデル+インドネシア語を英語に翻訳をすると, もっと良くなるかも? 

- nb19
    - ensembleをsimilarityの足し算とかで行うようにしてみた. 
    - これまでの方法だと(labelをつけてから和集合をとる), アンサンブルするにつれてlabelが単調増加してしまうので, こちらの方が良いのではないか. 
    - 足算して閾値をちょっと調整すると, cv = 0.8392くらいになった. 
    - 足し算の方がアンサンブルの効果が高そう. 

- nb20
    - 足し算でアンサンブルするようにしたバージョン. 
    - cvは上のnb19を参考にすると, cv = 0.8392くらい. 
    - tfidf+model2個で32-35分くらい. 

|th|lb|
|:--:|:--:|
|0.30|0.466|
|0.50|0.466|
|0.60|0.466|
|0.70|0.466|
|0.80|0.466|
|0.90|0.466| 

    - 全部同じになるなんてことがある? 
    - バグらせている気がする. 
    - それかthがもっと小さい方がよい? 
    - textはめっちゃ疎なベクトルになってそうだし. 

- nb21
    - bertのnotebookを写経する. 
    - まだbertについてよくわからないのでできない. 

**20200411**
- nb21
    - 動かせる形にしたので, 動かしてみる. 

- nb20
    - 足算してアンサンブルしたやつがすべてscore=0.466となってバグっている気がする. 

- nb08
    - どこを間違っているかみてたんだけど, これは確かに似てるよね...みたいなものを似ていると判断していて, textはそのまま採用して, 画像はかなり類似度が高い時のみ, 同じであると判定するようにアンサンブルする方が良い気がしてきた. 

**20200412**
- nb13
    - augmentationをちょっと強くしてCenterCropを入れたらほんの少しf1scoreがよくなった. 
    - cv = 0.76130...
    - image + tfidfのcv = 0.808284...
    - lb = 0.700
    - となった. 

- nb21
    - bertを動かした. 
    - forkしたnotebookはvalidationの切り方がよくないような気がしていて, きちんとgroupkfoldしたら, lossはあそこまで下がらなかった. 
    - imageよりlossは大きい. 
    - cv-f1scoreが全然だめ...

- nb16
    - nfnetの学習. 
    - lossがnanにならないことを祈る. 
    
- nb23
    - textでidを予測, imageでsimilarityを平均してidを予測, textとimageの予測をorでくっつけるようにしたらよかった. 

- distilationを試してみたい. 

**20200413**
- nb24
    - [cassava conpetition](https://speakerdeck.com/sansandsoc/diary-against-the-noisy-label)に載ってたnoisy labelに対処するSymmetric Lossを試してみたら, cv = 0.762..で改善した. 
    - effnetb4も試してみる. 

- 今日気づいたんだけど, 実は0.466はなにかのバグでpredが作られておらず, submission.csvがそのまま提出されているときに起こるバグなのかもしれない. 
- 途中でエラーになってもsubmission.csvがそのまま提出される仕様になっていたっぽい. 
- 必要メモリが大きすぎてmemory errorを起こした場合とか. 

**20200414**
- [このdiscussion](https://www.kaggle.com/c/shopee-product-matching/discussion/228794)が有用だった. 
    - distlibert
    - effnetb0b1b2みたいに小さめのモデルの方がいいらしい. 

- nb21
    - bertに再挑戦していた. 
    - max_len=128にするのと, 3epochごとにthresholdを最適化した. 
    - bertだとインドネシア語を翻訳とかした方がよい? 

**20200417**
- nb13
    - effcientnetb2を動かしていたがb2の方が良かった. 
    - これよく考えてみるとエポック数変えちゃったから単純に比較できないじゃん...
    - cv = 0.7743... 
    - lb = 0.708
    - tfidfこみでのcvは0.819607...
    - epoch数は重要なのか? 

- nb21
    - bertが全然うまくいっていなかった. 
    - 翻訳とかが必要だと思う.
    - discussionで紹介されていたインドネシア英語tranlaterを使ってみた。
    - とりあえず, 上の翻訳をする. 
    - あと多言語対応bertとか? 
    - それか[このnotebook](https://www.kaggle.com/tanulsingh077/metric-learning-pipeline-only-text-sbert)で紹介されているsbertとか? 

- アイデア
    - submit中にpseudo_labelingして, それを下にちょっとトレーニングするみたいなことをする
    - 時間がキツそう. 
    - pseudo_labelingするためにそもそも同じグループに属するものは同じにした方がよくねってなったので、dfsをして連結成分を取り出すことにした. -> nb26 

**20200418**
- アイデア
    - sbert
    - data cleansing 
    - nlpパートでstemmerとかもうちょっと工夫できるところがある
    - image -> nfnet,effnetv2
    - [noisy student](https://upura.hatenablog.com/entry/2020/02/18/180500), pseudolabeling的なことをする. 
    - BN層のweightを学習前半で固定する. 

- nb26
    - dfsした
    - この場合はthredholsを少し高めにとる方がcvが良かったので, lbもそれを参考にする. 

- nb11
    - nlpパートをする. 
    - インドネシア語の翻訳
    - stemmer
    - stopwordの除去
    - 絵文字の除去
    - してtfidfしてみる. 
    - 少し翻訳すると cv = 0.6330 -> cv = 0.631 となった. 
    - 邪魔そうなwordを除去すると, cv = 0.63316307656となった. 
    - もっとじゃまそうなものを削除すると cv = 0.64053...となった. -> 閾値調整でもっと上がるかも? 
    - 閾値を調整したら cv = 0.6684204..までいった.
    - そういえば上の翻訳すると少し悪くなるのはthの調整をしていないからかも? (sizeが小さくなるのでもっとthを小さくすべきということになる)
    - stopword除去+翻訳で cv = 0.6701くらいになった. 
    - stemmerは除去+stemmerで cv = 0.6511となった. 
    - wordの長さが3以下のものを除去するようにしたらもっとよくなった, cv = 0.6727...

- nb19
    - 上の処理をしてからtfidfをしたものをアンサンブルすると0.8397...まで上がった. 
    - 想像よりtextのthは小さくて良い? 

**20200419**
- アイデア
    - というかttaすべきだろ
    - もしかして温度パラメータって重要なパラメータ? 

- wordの前処理ありimageとアンサンブル. 
- 参考 : word前処理なし. th = (0.85,0.75) で 0.718くらい. 


|th|lb|
|:--:|:--:|
|(0.85,0.65)|0.686|
|(0.85,0.70)|0.699|
|(0.85,0.75)|0.705|
|(0.90,0.70)|0.700|
|(0.90,0.75)|0.700| 

- 前処理した方が悪くなった? 
- leakする要素は思いつかないけどな... 
- cvは良かったんだけどな... 

- word前処理なし, dfsありimageとアンサンブル

|th|lb|
|:--:|:--:|
|(0.85,0.75)|0.659|
|(0.90,0.85)|0.695|
|(0.90,0.80)|0.690|
|(0.95,0.85)|0.686|

- 以下はwordの前処理ありdfs, 

|th|lb|
|:--:|:--:|
|(0.85,0.80)|0.652|
|(0.85,0.75)|0.618|
|(0.90,0.75)|0.627|

- よくない. 

- nb27  
    - imageの類似度の指標としてknnがよく使われているっぽいのでそれを用いてみる. 
    - ほとんど変わらなかった. 当たり前か. 

- nb24
    - epochの前半でweightを固定したら, cv = 0.7725までよくなった. 

- nb28
    - training while submitを発動する. 
    - 

**20200420**
- 仮説
    - 実はモデルの精度のcvが良いのは過学習しているだけで(leakしてないと思うんだけどな...), lbでは悪くなる. 
    - 数字, #から始まる単語は商品コードを表している場合があって, そういうのは残したほうがいいらしい. 

**20200424**
- 上についてimagenet学習済みモデルだとまともなf1score = 0.41くらいしか行かないので学習はしている. 
- image 
- 今になって気づいたけどimagenetの転移学習をさせるならmean,stdを固定しないといけないらしい. 

- nb16
    - nfnetじゃなくてeca_nfnet_l0を使ったらlossがnanにならなくなった. (こっちの方がよく使われている?)

- nb11
    - textの前処理が本当に効果があるのか確かめる. 
    - 実はlbの閾値はもっと小さい説がある. 

**20200425**
- nb11
    - 閾値はもっと小さい方がよかった. 確認すべきだった. 
    - これまでのtextの前処理をしたやつのスコアは閾値が最適じゃないのであまり参考にならない. 
    - th = 0.65でlb = 0.602だった(textのみ). 

- nb24
    - imagenetのmean,stdで正規化をかけるようにしたら, cv = 0.7812になった. 

- nb19
    - textの前処理ときちんと学習させたb3でアンサンブルすると, cv = 0.84710となった. 
    - 前よりめっちゃいいじゃん. 
    - fold1も学習させたい. 

- nb30
    - textの閾値を調節したのとb3で提出したもの
    - cv = 0.847(fold0) 
    - imsizeを大きくする. 
    - これより下はsubmitしない. 

|th|lb|
|:--:|:--:|
|(0.80,0.70)||
|(0.85,0.65)|0.688|
|(0.85,0.70)||
|(0.90,0.65)|0.697|
|(0.90,0.70)|0.708|
|(0.90,0.75)||


- nb24
    - imsize = 512にしてb3をtrainingしてみた. 

**20200427** 
- nb30
    - b3でimsize=512のものとpreprocessなしのものを使うと, lb = 0.723で今のところ最高. 
    - preprocessしない方がlbは良いんだけどなんで? 
    - それに関するdiscussionをみて, 数字を消さないようにしたけどcvはよくなってlbは悪くなる. 
    - データサイズの問題か?

|th|lb|
|:--:|:--:|
|(0.85,0.80)||
|(0.90,0.75)|0.723|
|(0.90,0.80)||
|(0.90,0.85)||
|(0.95,0.80)||

- nb24
    - うえのimsize=512,b3がcv = 0.7997でimageだけみるとcvとlbは相関してるんだけど, textを入れるとなんか相関が取れない. 
    - textは全くデータの性質が違うっていう可能性もあるか. 

- nb31
    - もう一回ちゃんとedaする. 
    - labelを綺麗にする. (fold内でphashかlabelが一致したら, それを一緒のgroupにする.)
    - でtrainingしてみる. 

- ずっとbatchnormをfreezeする. 

**20200502**
- nb19
    - 最高cvは0.8565までよくなった. 

- nb31
    - データをちゃんとみる. 

- nb8
    - tfidfのcos　similarityが0なら消す
    - それだと消えてしまうものもある
    - サイズ表記が違ったら絶対同じにしない XXL,XL,M32,L30S38
    - 機種が違ったら絶対同じにしない(コード, 変な記号などから始まるもの,電子機器)
    - 2gramまでみて単位を考える.
    - 類義語が入っている場合は不可にする (ex mask,faceshield)
    - colorが違う
    - 上のことを試してみる!


























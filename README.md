# モバイル学習モデルの俯瞰: MobileNetV1からEfficientNetV2まで

![](https://raw.githubusercontent.com/yKesamaru/mobile_learning_models/master/assets/eye-catch.png)

- [モバイル学習モデルの俯瞰: MobileNetV1からEfficientNetV2まで](#モバイル学習モデルの俯瞰-mobilenetv1からefficientnetv2まで)
  - [モバイル学習モデル](#モバイル学習モデル)
  - [モバイル学習モデルの変遷](#モバイル学習モデルの変遷)
  - [パラメータ数とFLOPsについて](#パラメータ数とflopsについて)
  - [モデルの比較表](#モデルの比較表)
  - [一般的なサイズ感](#一般的なサイズ感)
  - [`timm`（PyTorch Image Models）について](#timmpytorch-image-modelsについて)
    - [timm の概要](#timm-の概要)
    - [主要な特徴](#主要な特徴)
  - [`timm`のモデルバリエーション](#timmのモデルバリエーション)
    - [`timm`で配布されている`MobileNetV3`モデルの一覧](#timmで配布されているmobilenetv3モデルの一覧)
    - [`timm`で配布されている`efficientnetv2`モデルの一覧](#timmで配布されているefficientnetv2モデルの一覧)
  - [MobileNetV3とEfficientNetV2の比較](#mobilenetv3とefficientnetv2の比較)
  - [入力サイズ](#入力サイズ)
    - [MobileNetV3の入力サイズ](#mobilenetv3の入力サイズ)
    - [EfficientNetV2の入力サイズ](#efficientnetv2の入力サイズ)
  - [参考文献](#参考文献)


## モバイル学習モデル
モバイル学習モデルは、モバイルデバイス上で動作する機械学習モデルを指します。これらは、リソースが限られた環境での高効率な運用を目的としています。

## モバイル学習モデルの変遷
**MobileNet**

* 2017年にGoogle AIの研究チームによって発表された、軽量で高速な画像認識モデル。
* "Depthwise Separable Convolution"を採用し、計算量を大幅に削減。
* ImageNetでのTop-1精度が70.6%と、軽量モデルとして高い精度を実現。
* スマートフォンやIoTデバイスなど、リソースが限られた環境での使用に適している。

**MnasNet**

* 2018年にGoogle AIによって発表された、MobileNetとは異なるアーキテクチャ。
* 自動化されたニューラルネットワーク設計（Neural Architecture Search, NAS）を用いて開発。
* ImageNetでのTop-1精度が74.0%と、更なる精度の向上を実現。

**EfficientNet**

* 2019年にGoogle AIによって発表された、MobileNetとMnasNetの研究成果を統合したモデル。
* スケーリング法則を用いてネットワークの幅、深さ、解像度を均等にスケーリングし、効率性を向上。
* ImageNetでのTop-1精度が84.6%と大幅な性能向上。

**EfficientNetV2**

* 2021年にGoogle AIによって発表された、EfficientNetの改良版。
* 畳み込み層の構造を最適化し、精度と効率性をさらに向上。
* ImageNetでのTop-1精度が86.1%。

**EfficientNetV3**

* EfficientNetV3の開発が進行中であるとの情報があるが、未確認。

**Vision Transformer**

* CNNとは異なり、純粋にTransformerベースのアーキテクチャを採用。
* 自己注意機構を主体とし、画像をパッチに分割してトランスフォーマーの入力として使用。

## パラメータ数とFLOPsについて
表に記載されているパラメータ数とFLOPsは参考値であり、モデルのバージョンや設定によって異なる場合がある。


## モデルの比較表

| 項目 | MobileNet | MnasNet | EfficientNet | EfficientNetV2 |
|---|---|---|---|---|
| 発表年 | 2017 | 2018 | 2019 | 2021 |
| アーキテクチャ | 畳み込み層とプーリング層の組み合わせ | 畳み込み層とモジュール化されたアテンションの組み合わせ[^0] | 畳み込み層とモジュール化されたアテンションの組み合わせ | 畳み込み層とモジュール化されたアテンションの組み合わせ[^0] |
| 特徴 | 軽量で高速 | 軽量で高速 | 高精度で軽量 | 高精度で軽量 |
| パラメータ数 | 4.2M | 1.9M | 1.3M | 2.7M |
| FLOPs | 560M | 1.2G | 1.2G | 2.4G |
| 精度 | ImageNetでのTop-1精度: 約70%~73% | ImageNetでのTop-1精度: 74.0% | ImageNetでのTop-1精度: 約77%~84.6% | ImageNetでのTop-1精度: 約85%~88% |

[^0]: 「畳み込み層とモジュール化されたアテンションの組み合わせ」という表現は、畳み込みニューラルネットワーク（CNN）のアーキテクチャに、アテンションメカニズムを組み合わせたものを指します。このアテンションメカニズムは、Transformerで使用されるアテンションと同じ根本的な概念に基づいています。
アテンションメカニズムは、ニューラルネットワークが入力データの特定の部分に「注目」を集中させることを可能にする技術です。Transformerモデルで使われるアテンションは、特に「自己アテンション」と呼ばれ、入力データの異なる位置間の関連性をモデル化します。
畳み込み層は、画像などの空間データを処理するのに適しているため、多くのCNNアーキテクチャで広く使用されています。
一方で、アテンションメカニズムは、データの重要な部分に焦点を当て、コンテキスト情報をより効果的に取り入れるのに役立ちます。
畳み込み層とアテンションを組み合わせることで、ネットワークは局所的な特徴（畳み込み層によって捉えられる）とグローバルな依存関係（アテンションによって捉えられる）の両方を効果的に利用できるようになります。
EfficientNetやその他の最新のニューラルネットワークアーキテクチャでは、このような組み合わせが採用されており、特に画像分類タスクにおいて高い性能を発揮しています。アテンションを使用することで、ネットワークは画像の重要な部分に焦点を合わせ、全体のコンテキストに基づいてより正確な予測を行うことができます。

**項目の説明**

* 発表年: モデルが発表された年
* アーキテクチャ: モデルの構成
* 特徴: モデルの特徴
* パラメータ数: モデルの重みパラメータの数
* FLOPs: モデルの演算量（各層で実行される演算の数の合計の目安）
* 精度: モデルの精度

※パラメータ数については[こちら: OpenAIが発見したScaling Lawの秘密](https://deeplearning.hatenablog.com/entry/scaling_law)を参照してください。
※FLOPsについては[こちら: 機械学習モデルのトレーニング時演算量の推移](https://hacarus.com/ja/ai-lab/03102022-tech-blog-18/)を参照してください。

## 一般的なサイズ感
| モデル | パラメータ数 | モデルサイズ (MB) |
|---|---|---|
| efficientnetv2-s | 1.2M | 4.4 |
| efficientnetv2-m | 2.2M | 7.2 |
| efficientnetv2-l | 4.0M | 10.5 |
| MobileNet v3-Small | 5.3M | 2.9 |
| efficientnetv2-b0 | 5.3M | 2.9 |
| MobileNet v3-Medium | 11.2M | 4.2 |
| efficientnetv2-b1 | 11.2M | 4.2 |
| MobileNet v3-Large | 22.2M | 7.1 |
| efficientnetv2-b2 | 22.2M | 7.1 |


## `timm`（PyTorch Image Models）について
### timm の概要
- **目的と機能**: `timm`は、画像認識に関連する様々な最新のニューラルネットワークモデルをPyTorchで使えるようにするライブラリです。これには、事前訓練済みモデルの提供、新しいモデルアーキテクチャの実験、既存のモデルの微調整やカスタマイズが含まれます。

### 主要な特徴
1. **多様なモデル**: `timm`は、EfficientNet、ResNet、DenseNetなど、多くの人気モデルを含む事前訓練済みモデルを提供します。

2. **事前訓練済みの利便性**: 多くのモデルがImageNetなどの大規模データセットで事前訓練されており、これによりユーザーは迅速にこれらのモデルを活用して、自分のデータセットでの学習や推論を行うことができます。

3. **カスタマイズと実験**: `timm`は、モデルのアーキテクチャをカスタマイズしたり、新しいアーキテクチャを試したりするのにも適しています。

## `timm`のモデルバリエーション
MobileNet v3とEfficientNetV2のパラメータ数とモデルサイズは、以下のとおりです。

### `timm`で配布されている`MobileNetV3`モデルの一覧

| Model name | Params (M) | Image size: train, test |
|------------|------------|-------------------------|
| [mobilenetv3_large_100.miil_in21k](https://huggingface.co/timm/mobilenetv3_large_100.miil_in21k) | 18.6 | 224 x 224 |
| [mobilenetv3_large_100.miil_in21k_ft_in1k](https://huggingface.co/timm/mobilenetv3_large_100.miil_in21k_ft_in1k) | 5.5 | 224 x 224 |
| [mobilenetv3_large_100.ra_in1k](https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k) | 5.5 | 224 x 224 |
| [mobilenetv3_rw.rmsp_in1k](https://huggingface.co/timm/mobilenetv3_rw.rmsp_in1k) | 5.5 | 224 x 224 |
| [mobilenetv3_small_050.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_050.lamb_in1k) | 1.6 | 224 x 224 |
| [mobilenetv3_small_075.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_075.lamb_in1k) | 2.0 | 224 x 224 |
| [mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k) | 2.5 | 224 x 224 |
| [tf_mobilenetv3_large_075.in1k](https://huggingface.co/timm/tf_mobilenetv3_large_075.in1k) | 4.0 | 224 x 224 |
| [tf_mobilenetv3_large_100.in1k](https://huggingface.co/timm/tf_mobilenetv3_large_100.in1k) | 3.9 | 224 x 224 |
| [tf_mobilenetv3_large_minimal_100.in1k](https://huggingface.co/timm/tf_mobilenetv3_large_minimal_100.in1k) | 3.9 | 224 x 224 |
| [tf_mobilenetv3_small_075.in1k](https://huggingface.co/timm/tf_mobilenetv3_small_075.in1k) | 2.0 | 224 x 224 |
| [tf_mobilenetv3_small_100.in1k](https://huggingface.co/timm/tf_mobilenetv3_small_100.in1k) | 2.5 | 224 x 224 |
| [tf_mobilenetv3_small_minimal_100.in1k](https://huggingface.co/timm/tf_mobilenetv3_small_minimal_100.in1k) | 2.0 | 224 x 224 |

### `timm`で配布されている`efficientnetv2`モデルの一覧
| Model name                                                                                         | Params (M) | Image size: train, test          |
|----------------------------------------------------------------------------------------------------|------------|-----------------------------------|
| [efficientnetv2_rw_s.ra2_in1k](https://huggingface.co/timm/efficientnetv2_rw_s.ra2_in1k)      | 23.9       | 288 x 288, 384 x 384              |
| [efficientnetv2_rw_t.ra2_in1k](https://huggingface.co/timm/efficientnetv2_rw_t.ra2_in1k)      | 13.6       | 224 x 224, 288 x 288              |
| [gc_efficientnetv2_rw_t.agc_in1k](https://huggingface.co/timm/gc_efficientnetv2_rw_t.agc_in1k) | 13.7       | 224 x 224, 288 x 288              |
| [tf_efficientnetv2_b0.in1k](https://huggingface.co/timm/tf_efficientnetv2_b0.in1k)             | 7.1        | 192 x 192, 224 x 224              |
| [tf_efficientnetv2_b1.in1k](https://huggingface.co/timm/tf_efficientnetv2_b1.in1k)             | 8.1        | 192 x 192, 240 x 240              |
| [tf_efficientnetv2_b2.in1k](https://huggingface.co/timm/tf_efficientnetv2_b2.in1k)             | 10.1       | 208 x 208, 260 x 260              |
| [tf_efficientnetv2_b3.in1k](https://huggingface.co/timm/tf_efficientnetv2_b3.in1k)             | 14.4       | 240 x 240, 300 x 300              |
| [tf_efficientnetv2_b3.in21k](https://huggingface.co/timm/tf_efficientnetv2_b3.in21k)           | 46.4       | 240 x 240, 300 x 300              |
| [tf_efficientnetv2_b3.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_b3.in21k_ft_in1k) | 14.4       | 240 x 240, 300 x 300              |
| [tf_efficientnetv2_l.in1k](https://huggingface.co/timm/tf_efficientnetv2_l.in1k)               | 118.5      | 384 x 384, 480 x 480              |
| [tf_efficientnetv2_l.in21k](https://huggingface.co/timm/tf_efficientnetv2_l.in21k)             | 145.2      | 384 x 384, 480 x 480              |
| [tf_efficientnetv2_l.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_l.in21k_ft_in1k) | 118.5      | 384 x 384, 480 x 480              |
| [tf_efficientnetv2_m.in1k](https://huggingface.co/timm/tf_efficientnetv2_m.in1k)               | 54.1       | 384 x 384, 480 x 480              |
| [tf_efficientnetv2_m.in21k](https://huggingface.co/timm/tf_efficientnetv2_m.in21k)             | 80.8       | 384 x 384, 480 x 480              |
| [tf_efficientnetv2_m.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k) | 54.1       | 384 x 384, 480 x 480              |
| [tf_efficientnetv2_s.in1k](https://huggingface.co/timm/tf_efficientnetv2_s.in1k)               | 21.5       | 300 x 300, 384 x 384              |
| [tf_efficientnetv2_s.in21k](https://huggingface.co/timm/tf_efficientnetv2_s.in21k)             | 48.2       | 300 x 300, 384 x 384              |
| [tf_efficientnetv2_s.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k) | 21.5       | 300 x 300, 384 x 384              |
| [efficientnetv2_rw_m.agc_in1k](https://huggingface.co/timm/efficientnetv2_rw_m.agc_in1k)       | 53.2       | 320 x 320, 416 x 416 |
| [tf_efficientnetv2_xl.in21k](https://huggingface.co/timm/tf_efficientnetv2_xl.in21k)           | 234.8      | 384 x 384, 512 x 512 |
| [tf_efficientnetv2_xl.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_xl.in21k_ft_in1k) | 208.1      | 384 x 384, 512 x 512 |

## MobileNetV3とEfficientNetV2の比較
MobileNetV3とEfficientNetV2は、どちらも軽量なアーキテクチャでありながら、優れた精度を実現できるCNNです。
以下に両者の比較として、いくつかの共通点と相違点を挙げます。

**共通点**

* どちらも、[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]と[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]を採用している。
* どちらも、複数のバリエーションが用意されている。

[^1]: チャンネルごとに畳み込みを行う畳み込み層の一種です。通常の畳み込み層では、各チャネル間で畳み込みが行われますが、Depthwise Convolutionでは、各チャネルごとに独立して畳み込みが行われます。これにより、計算量を大幅に削減ができます。

[^2]: 各チャネルの重要度を調整するモジュールです。まず、チャネルごとに出力を平均化します。次に、平均化された出力を2つの全結合層に通して、各チャネルの重要度を算出します。最後に、重要度に基づいて各チャネルの出力をスケーリングします。これにより、各チャネルの重要度に応じて出力を調整ができます。

[^3]: MobileNetV3で採用されている基本的な構成単位であるブロックです。3x3のDepthwise Convolutionを1x1のConvolutionで挟んだ構造になっています。Depthwise Convolutionで特徴量を抽出し、1x1のConvolutionでチャンネル数を調整します。

[^4]: EfficientNetV2で採用されている基本的な構成単位であるブロックです。3x3の畳み込み層とSqueeze-and-Excitation Moduleで構成されています。畳み込み層で特徴量を抽出し、Squeeze-and-Excitation Moduleで各チャネルの重要度を調整します。

[^5]: 畳み込み層を2つの小さな畳み込み層に分解する手法です。通常の畳み込み層では、フィルターサイズと同じ大きさのカーネルを用いて畳み込みが行われますが、Factorized Convolutionでは、フィルターサイズを半分にすることで、カーネルを2つに分解ができます。これにより、計算量を削減ができます。

**相違点**

* **アーキテクチャの構成**
    * MobileNetV3は、[Inverted Residual Block](https://arxiv.org/abs/1801.04381)[^3]を基本的な構成単位としています。
    * EfficientNetV2は、[Squeeze-and-Excitation Block](https://arxiv.org/abs/1905.11175)[^4]を含み、スケーラビリティと効率を重視した構造を採用しています。

* **計算量の削減手法**
    * MobileNetV3は、[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]と[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]によって計算量を削減します。
    * EfficientNetV2は、[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]と[Factorized Convolution](https://arxiv.org/abs/1603.08024)[^5]によって計算量を削減します。

* **パラメータ数**
    * MobileNetV3は、EfficientNetV2よりもパラメータ数が少ない。

* **精度**
    * 一般的に、EfficientNetV2の方がMobileNetV3よりも精度が高い。

**具体的な比較**

**アーキテクチャの構成**

MobileNetV3の[Inverted Residual Block](https://arxiv.org/abs/1801.04381)[^3]は、3x3の[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]を1x1のConvolutionで挟んだ構造です。[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]は、チャンネルごとに畳み込みを行うため、計算量を大幅に削減できます。1x1のConvolutionは、チャンネル数を調整するために使用されます。

EfficientNetV2の[Squeeze-and-Excitation Block](https://arxiv.org/abs/1905.11175)[^4]は、3x3の畳み込み層と[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]で構成されています。[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]は、各チャネルの重要度を調整するために使用されます。

**計算量の削減手法**

MobileNetV3は、[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]と[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]によって計算量を削減します。[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]は、チャンネルごとに畳み込みを行うため、計算量を大幅に削減できます。[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]は、各チャネルの重要度を調整することで、不要なチャネルの処理を省略できます。

EfficientNetV2は、[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]と[Factorized Convolution](https://arxiv.org/abs/1603.08024)[^5]によって計算量を削減します。[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]は、MobileNetV3と同様に、各チャネルの重要度を調整することで、不要なチャネルの処理を省略できます。[Factorized Convolution](https://arxiv.org/abs/1603.08024)[^5]は、畳み込み層を2つの小さな畳み込み層に分解することで、計算量を削減します。

**パラメータ数**

MobileNetV3は、EfficientNetV2よりもパラメータ数が少ない傾向にあります。これは、MobileNetV3が[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]と[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]を採用しているためです。[Depthwise Convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)[^1]は、チャンネルごとに畳み込みを行うため、パラメータ数が少ないという特徴があります。[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]も、パラメータ数が少ないという特徴があります。

**精度**

一般的に、EfficientNetV2の方がMobileNetV3よりも精度が高い傾向にあります。これは、EfficientNetV2が[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]と[Factorized Convolution](https://arxiv.org/abs/1603.08024)[^5]を採用しているためです。[Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507)[^2]は、各チャネルの重要度を調整することで、精度の向上に寄与します。[Factorized Convolution](https://arxiv.org/abs/1603.08024)[^5]は、計算量を削減しながらも、精度の低下を抑えることができます。

具体的な用途としては、MobileNetV3は、低スペックのデバイスで動作するアプリケーションに適しています。EfficientNetV2は、精度を重視するアプリケーションに適しています。

## 入力サイズ
### MobileNetV3の入力サイズ
- **固定サイズ**: MobileNetV3は、一般的に224x224ピクセルの固定サイズの入力を想定しています。これは、モデルの設計が特定の入力サイズに最適化されているためです。
- **軽量化と効率性**: MobileNetV3は、主にモバイルデバイスやリソース制約のある環境での使用を目的として設計されています。固定サイズの入力を使用することで、モデルの軽量化と効率性を保っています。

### EfficientNetV2の入力サイズ
- **スケーラブルなアーキテクチャ**: EfficientNetV2は、スケーラブルなアーキテクチャを採用しています。これにより、同じモデルファミリー内でさまざまなサイズのバリエーションが可能となり、小さいモデルから大きなモデルまで様々なリソース要件に対応できます。
- **[Compound Scaling](https://qiita.com/omiita/items/83643f78baabfa210ab1#3-compound-model-scaling%E8%A4%87%E5%90%88%E3%83%A2%E3%83%87%E3%83%AB%E3%82%B9%E3%82%B1%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0)[^6]**: EfficientNetV2は、ネットワークの深さ、幅、および解像度のバランスをとるための「[Compound Scaling](https://qiita.com/omiita/items/83643f78baabfa210ab1#3-compound-model-scaling%E8%A4%87%E5%90%88%E3%83%A2%E3%83%87%E3%83%AB%E3%82%B9%E3%82%B1%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0)[^6]」という手法を使用しています。これにより、モデルは異なる入力サイズに効率的に対応できるようになります。
- **柔軟性と性能の最適化**: 異なる入力サイズをサポートすることで、EfficientNetV2は、特定のタスクや利用可能な計算リソースに基づいて性能を最適化できます。



[^6]: Compound Scalingは、EfficientNetの核となるコンセプトで、モデルのサイズ、精度、および計算効率の間の長年のトレードオフを解決するために開発されました。この方法の背後にあるアイデアは、ニューラルネットワークの三つの重要な次元、すなわち幅（width）、深さ（depth）、および解像度（resolution）をスケーリングすることです​​。Compound Scalingの具体的な手法には、ユーザーが定義するcompound coefficient（ϕ）を使用して、ネットワークの幅、深さ、解像度を統一的にスケールアップします。例えば、入力画像が大きい場合、ネットワークはより多くの層を必要とし、より細かいパターンをキャプチャするためにより多くのチャネルが必要になります。この直感に基づき、Compound Scalingはこれらの次元を一様にスケーリングすることで、モデルの精度と効率を一貫して向上させます​​​​。このスケーリング方法は、従来のスケーリング方法と比較して、MobileNet（+1.4% ImageNet精度）やResNet（+0.7%）などの既存モデルの精度と効率を一貫して向上させることが示されています。Compound Scalingの有効性は、基本となるネットワークアーキテクチャにも大きく依存しています​​。

## 参考文献
- 記事
  - [MobileNet(v1,v2,v3)を簡単に解説してみた](https://qiita.com/omiita/items/77dadd5a7b16a104df83)
  - [2019年最強の画像認識モデルEfficientNet解説](https://qiita.com/omiita/items/83643f78baabfa210ab1)
  - [2021年最強になるか！？最新の画像認識モデルEfficientNetV2を解説](https://qiita.com/omiita/items/1d96eae2b15e49235110)
- 動画
  - [【画像認識の代表的なモデル#12】EfficientNet（2019）](https://www.youtube.com/watch?v=2Ggvq7TJx_8)
  - [backboneとしてのtimm入門](https://www.youtube.com/watch?v=YeyK1QhEB6A)

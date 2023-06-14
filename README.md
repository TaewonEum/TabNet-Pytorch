# TabNet-Pytorch

참고 논문:  https://arxiv.org/pdf/1908.07442.pdf

# Concept 
        
딥러닝은 이미지, 자연어 등 비정형 데이터를 처리하는데 좋은 성과를 보이지만 정형 데이터는 관련 분야에서 그러지 못하는 상황임

하지만 TabNet 기법을 통해 별도의 전처리 과정 없이 정형데이터를 최적의 알고리즘으로 학습할 수 있음

# Tabnet Architecture

![image](https://github.com/eumtaewon/TabNet-Pytorch/assets/104436260/013b6215-728c-4f4f-99de-8b39bc6f9936)

위 그림과 같이 Input 변수들로 workclass, education, occupation등 다양한 변수들이 선택된다

변수가 선택되면 Processing 과정을 거치게 된다.

이러한 Step 순차적으로 거쳐 Output을 도출하게 된다.

Step은 변수 선택과 변수 전처리 과정으로 이루어진다.

각 스텝은 특정 변수에 집중하고, 이는 모델이 각 변수의 중요성을 학습하는 데 도움이 된다.

한 Step마다 고려할 변수를 선택한 후 neural net으로 처리한다.

위 그림을 예시로 설명을 진행해보면

예시는 Adult Census Income 데이터 셋을 사용하여 한 row당 소득 수준을 예측하는 모델을 TabNet을 통해 생성하는 과정을 보여준다.

그림에서는 두개의 decision step이 묘사되어 있다 첫 번째 Step에서는 직업과 관련된 변수를 선택하여 처리하고, 두 번째 Step에서는

투자와 관련된 변수를 처리하고 있다.

최종적으로 두 Step이 모두 끝난 다음 output을 aggregate하여 최종 결과값을 도출한다.

해당 그림의 결과값으로는 해당 row가 50,000$의 소득 이상인지를 분류하는 binary classification값이 도출된다.

전체적인 과정을 보면 TabNet이 decision tree의 특징을 많이 갖고 있다.

첫 번째로 TabNet은 sparse instance-wise feature selection, 즉 각 data point 별로 필요한 feature만 선택하여 처리한다.

어떠한 feature를 선택할지는 데이터로부터 학습한다. 

두 번째로 TabNet은 sequential multi-step 구조를 갖고 있으며 모든 step의 output을 고려하여 최종 결정을 내리는 방식으로 앙상블과 같은 효과를 낸다. 

마지막으로 선택한 feature의 non-linear 처리를 통해 풍부한 함수 표현이 가능하다.

# Unsupervised pre-training, Supervised fine-tuning

## TabNet Unsuprervised pre training

![image](https://github.com/eumtaewon/TabNet-Pytorch/assets/104436260/daa5f8c3-de2c-4217-9c70-3d58acf0d436)

비지도 학습은 레이블이 없는 데이터에서 유의미한 특성을 학습하는 방법입니다.

일부 값들을 masking하고 모델이 이를 예측하도록 합니다. Tabnet은 변수 간의 상호 의존성을 학습합니다. 

일부 마스킹된 정형 데이터를 Tabnet Encoder 부분의 입력으로 받아 여기서 다양한 변수들의 복잡한 상호작용을 학습합니다.

Encoder에서는 일부 변수만 선택하여 처리합니다. 이를 통해 고차원 정형 데이터에서 모델이 가장 중요한 변수에 집중할 수 있도록 도움을 줍니다.

TabNet Decoder 에서는 Encoder가 생성한 잠재 표현을 받아 최종 예측값을 생성함. 

즉 Encoder에서 입력데이터 처리->중요변수 선택->복잡한 패턴 학습->학습된 정보 Decoder로 이동->최종 예측값 출력

## Tabnet supervised

![image](https://github.com/eumtaewon/TabNet-Pytorch/assets/104436260/6c426888-2c9a-4bb3-b948-ca6c60c8e855)

TabNet은 먼저 자기 지도 학습을 통해 테이블 데이터의 복잡한 패턴을 학습하고, 

이를 바탕으로 예측 작업을 수행하여 감독 학습의 성능을 향상시키는 방식을 채택하고 있습니다.

Tabnet Encoder 부분은 Unsupervised pre training과 같다

Decision Masking은 TabNet의 또 다른 핵심 요소로 입력 특성의 어떤 부분이 예측에 중요한지를 결정한다.

Encoder가 출력한 각 특성의 중요도를 계산하고 이를 기반으로 중요한 특성을 선택한다.

이렇게 선택된 특성들만이 다음 Decision Step으로 전달되고 나머지 특성들은 제외된다.

이를 통해 중요한 변수만을 학습할 수 있게 된다.

TabNet의 supervised fine-tuning 과정에서는 이러한 TabNet Encoder와 Decision Masking이 핵심적으로 사용됩니다. 

처음에는 unsupervised learning을 통해 데이터의 복잡한 패턴을 학습하고, 그 다음에는 이를 fine-tune하여 특정 예측 작업을 수행하도록 학습합니다. 

이 과정에서 Encoder와 Decision Masking은 데이터의 중요한 특성을 선택하고, 이를 바탕으로 예측을 수행하는 핵심적인 역할을 수행합니다.

# TabNet Network

![image](https://github.com/eumtaewon/TabNet-Pytorch/assets/104436260/f31c5448-f712-4685-bee5-3419be173bb7)

## (a) TabNet encoder

- TabNet encoder는 feature transformer, attentive transformer, feature masking으로 구성되어 있음

- Feature transformer는 입력 특성을 처리하고 변형한다->비선형적 변환

- Attentive transformer는 어떠한 변수가 중요한지를 결정하는 역할을 함, 모델은 각 단계에서 중요한 변수에 가중치를 줘서 더 많은 학습을 하게 하고 중요하지 않은 변수는 가중치를 적게주어 학습이 더 안되게 함

- feature masking에서는 특정 변수를 선택하거나 선택하지 않는 방식으로 작동함. 모델이 특정 변수에 지나치게 과적합되어 있다면 해당 변수를 마스킹 처리하여 나머지 변수로 모델을 학습하여 모델의 일반화 성능을 향상시킴


# TabNet-Pytorch

참고 논문:  https://arxiv.org/pdf/1908.07442.pdf

# Concept 
        
        -정형 데이터에서 주로 사용하는 딥러닝 모델
        
        정형 데이터는 주로 Gradient Boosting 알고리즘을 사용한 예측 모델을 많이 사용함
        
        XGBoost, LightGBM, CatBoost
        
        일반적으로 정형 데이터는 대략적인 초평면 경계를 가지는 Mainfold라고 함.
        
        부스팅 모델들은 이러한 Mainfold에서 결정할때 더욱 효율적으로 작동함.
        
        하지만 정형 데이터가 많다면  Multi-modal Learning을 통해 정형 데이터와 비정형 데이터를 학습에 함께 사용할 수 있음
        
        
# TabNet Advantages


        - 전처리 필요없이 raw한 데이터를 입력으로 사용 가능함
        
        - 각 의사결정에서 어떤 feature를 사용할지를 선택함
        
# Architecture

![image](https://user-images.githubusercontent.com/104436260/233883575-2b54b2d8-9d6f-4643-907e-79ffeb770ad1.png)

        - Step 1~N으로 나누어져있음
        
        - 각 Step 마다 Feature transformer(변수 변환)
        
        - Attentive transformer(주어진 입력에 대해 정보를 추출)
        
        - Feature masking(불필요한 변수 제거, 즉 모델에서 사용할 변수를 선택해주는 selection 단계)으로 구성됨
        
        - Split block에서 Feature transformer해준 데이터를 두개로 나누어 줌
        
        -하나는 비선형함수에 적용하여 최종 아웃풋 data로 만들고
        
        - 나머지 하나는 다음 Attentive transformer로 넘겨준다.
        
        그 후, Feature를 selection하는 Mask block은 각 Step에서 Feature가 작동하는 것에 대한 Insight를 제공할 수 있고, 
        Agg(regate) Block을 통해 궁극적으로는 어떤 Feature가 중요한지에 대한 것을 알 수 있습니다.
        
        
# Input data layer


        - 기본적으로 정형 데이터는 수치형 변수로 이루어져 있을 것이다. 그 중 연속형(이산형) 변수, 범주형 변수로 이루어져 있을 것 이다.
        
        범주형 변수 즉(0,1,2,3,4,5)와 같이 코딩이 되어있는 변수들은 One-hot encoding으로 처리를 해주어야 딥러닝 모델에 적용시킬 수 있다.
        
        - Batch Normalization을 통해 정규화를 대체함
        
        즉 연속형(이산형)->정규화 해주고, 범주형 변수는 임베딩해주어 Feature transformer에 들어감

# Feature transformer Architecture

![image](https://user-images.githubusercontent.com/104436260/233886690-89eef217-4919-4e58-afd7-68f9cc82c64a.png)       

        -Feature Transformer layer가 수행하는 활동은 입력데이터에 대한 전처리(표준화, 정규화)
        - Batch Normalization-> 각 컬럼에 대한 정규화 진행
        -Gated Linear Unit(원래 정보를 유지하기 위한 Resudual connection과 
        Sigmoid함수를 통해 선택적으로 유의미한 정보 추출하여 둘을 곱함)=>이게 output임

        - Fully Connected-> Batch Normalization -> Gated Linear Unit를 4번 반복하는 구조임
        
        - 기본적으로 이미지 딥러닝에서 Fully Connected는 다차원의 텐서 데이터를 일차원 벡터형태로 변환해주는 층임
        하지만 정형데이터에서는 각 컬럼이 하나의 차원에 해당됨. 이 차원을 줄이거나 유지하여 모델에 적용시켜주는 기능을 FC가 수행함
         
        - Gated Linear Unit-> 입력 데이터의 정보를 필요한 부분만 선택적으로 전달하기 위한 기능을 수행하는 층
        중요한 정보만을 선택하고자 할때 활용->회귀분석의 Stepwise, backward와 같은 변수선택이라고 보면 될 것 같음
        
        Resnet과 같이 원래정보를 유지한 데이터 activation function을 거쳐서 나온 유의미한 데이터를 element wise하고 그 데이터를 다시 
        fully connected -> Batch Normalization-> Gated Linear unit을 해줌
        

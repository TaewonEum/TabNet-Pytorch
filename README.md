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
        
        




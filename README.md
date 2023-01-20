# 2021_korea_AI
2021 국립국어원 언어능력 평가 대회 3등 솔루션 입니다. 

기존 저장소가 삭제되어 ppt를 확인해주세요

## dev setup
1. pytorch는 다음과 같이 터미널에서 실행해 주세요 
    ```
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    ```
    
2. requirements.txt 에 있는것들 install 
    ```
    pip install -r requirements.txt
    ```

## fine_tuning code 
{task_name}_fine_tune.py   
-> 각 task 마다 fine-tuning 하는 코드 

## inference(submission) code 
{task_name}_submission.py, concat_submission   
-> json_sample.json(대회 제출 양식)에 맞추기 위한 코드 

# KHDV
Code thực nghiệm KHDV
## Bài báo thực nghiệm :  SpokenWOZ: A Large-Scale Speech-Text Dataset for Spoken Task-Oriented Dialogue in Multiple Domains

- Tác giả: Shuzheng Si, Wentao Ma, Haoyu Gao, Yuchuan Wu, Ting-En Lin, Yinpei Dai, Hangyu Li, Rui Yan, Fei Huang and Yongbin Li

- Arxive:  [Link](https://arxiv.org/abs/2305.13040)  

## Dataset

- Dataset: https://spokenwoz.github.io/SpokenWOZ-github.io/
- - 5,700 dialogues ranging from single-domain to multi-domain in SpokenWOZ. 

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "dialogue_act": {
        "$act_name": [
          [
            "$slot_name",
            "$action_value"
          ]
        ]
      },
      "span_info": [
        [
          "$act_name"
          "$slot_name",
          "$action_value"
          "$start_charater_index",
          "$exclusive_end_character_index"
        ]
  }
}
```

The ASR transcription for each dialogue is recorded in the "words" field in every turn.  

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "words": [
        {
        "$word_context": "$word",
        "$begin_time": "$begintime",
        "end_time": "$endtime",
        "channel_id": "$channel",
        "word_index": "$index",
        }
  }
}
```

### LLMs

**Environment Setup**

```
pip install openai
```

#### DST

Please place the text data of SpokenWOZ in  `./LLM/dst/` 

```
python dst_data.py
(modify the openai key in the python file) sh run_chatgpt.sh / run_text003.sh
(modify the file dir in the python file) python eval_JPA.py
```

#### Response Generation

Please place the text data of SpokenWOZ in  `./LLM/response/`. Make sure you get the dst results, then place the prediction file in this folder.

```
sh chatgpt.sh
sh text_003.sh
```




### Finetuning

Fo supervised models, please kindly find more details from their original improvements.

#### SPACE+TripPy and SPACE+WavLM+TripPy 

**Environment Setup**

```
pip install -r requirement.txt
```

**Training & Evaluation**

Here `data.sh` does the processing of the audio file and processes it to`./trippy/../audio/` (will be used by subsequent models) 

Meanwhile, please place the text data of SpokenWOZ in  `./trippy/data` 

```
cd ./trippy
sh scripts/data.sh
sh scripts/train.sh
```



#### UBAR 

**Environment Setup**

```
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

**Training**

Please place the text data of SpokenWOZ in `./ubar/data/multi-woz` and move db folder and ontology.json to  `./ubar/db`

```
cd ./ubar
sh data_process.sh
sh train_dst.sh
sh train_response.sh
```

**Evaluation**

**Dialog State Tracking**

```
path='YOUR_EXPERIMENT_PATH'
python train_DST.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=False use_true_prev_aspn=True use_true_prev_resp=True use_true_db_pointer=False
```

**Policy Optimization (Act and Response Generation)**

```
path='YOUR_EXPERIMENT_PATH'

python train.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=True use_true_prev_aspn=False use_true_db_pointer=True use_true_prev_resp=False use_true_curr_bspn=True use_true_curr_aspn=False use_all_previous_context=True cuda_device=0
```

**End-to-end Modeling (Belief state, Act and Response Generation)**

```
path='YOUR_EXPERIMENT_PATH'
python train.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=False use_true_prev_aspn=False use_true_db_pointer=False use_true_prev_resp=False use_true_curr_bspn=False use_true_curr_aspn=False use_all_previous_context=True cuda_device=0
```



#### GALAXY, SPACE, SPACE+WavLM and SPACE+WavLM$_{align}$

**Environment Setup**

```
pip install -r requirement.txt

```

**Training & Evaluation**

Processing of text data:

```
sh data_process.sh
sh move_data.sh
```

The baseline contains the main following resources:

- `space-3`: It contains GALAXY, SPACE.
- `space_concat`:It contains SPACE+WavLM.
- `space_word`: It contains  SPACE+WavLM$_{align}$.

For SPACE and GALAXY, you need to download the corresponding models and place them in `. /space_baseline/space_word/space/model/`

```
(Select the target model)cd space-3 or space_concat or space_word

(Optional) sh train_galaxy.sh
sh train_space.sh
```




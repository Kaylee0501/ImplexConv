# ImplexConv
<p align="center">
  <a href="https://arxiv.org/abs/2503.07018"><img src="https://img.shields.io/badge/📝-Paper-blue" height="23"></a>
  <a href="https://huggingface.co/datasets/Kaylee0501/ImplexConv" ><img src="https://img.shields.io/badge/🤗-Data-green" height="23"></a>
</p>

 We design a large-scale multi-session conversation dataset to study implicit reasoning in personalized conversations and a hierarchical tree framework for for efficient, level-based retrieval.

 ## Installation
```bash
conda create -n ImplexConv python=3.9
conda activate ImplexConv
python -m pip install -r requirements.txt
```
If you need to use OpenAI APIs, you will need to obtain an API key [here](https://beta.openai.com/). 
```
export OPENAI_API_KEY=[your OpenAI API Key]
```

## Dataset
All datasets referenced in the paper are available [on HuggingFace](https://huggingface.co/datasets/Kaylee0501/ImplexConv).


## Usage
1. Create conversation summarization and facts:
```
python fact_sum_batch.py \
    --home_dir ./datasets \
    --dataset_name opposed_reasoning \
    --model_type gpt-4o-mini \
    --output_file summarized_opposed_facts.json
```
3. Generate the response and retrieved content:
```
python fact_topic_reasoning.py \
    --home_dir ./datasets \
    --dataset_name opposed_reasoning \
    --model_type gpt-4o-mini \
    --summy_info summarized_opposed_facts.json \
    --output_response_file opposed_response.json \
    --output_retrieve_file opposed_retrieved_text.json
```

## Citation
If you find the work useful, please cite:

```
@article{li2025toward,
  title={Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning},
  author={Li, Xintong and Bantupalli, Jalend and Dharmani, Ria and Zhang, Yuwei and Shang, Jingbo},
  journal={arXiv preprint arXiv:2503.07018},
  year={2025}
}
```

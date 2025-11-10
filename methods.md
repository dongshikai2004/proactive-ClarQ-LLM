## prepare

```bash
conda create -n clarq python==3.10
pip install -r requirement.txt
```

add API_KEYS in `ALL_KEYS`

```python
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
```

## run

### create conversations

```bash
python l2l.py --seeker_agent_llm qwen3-8b --provider_agent_llm gemini-2.5-flash --task_data_path data/English 
```

### evaluate

```bash
python evaluation.py gemini-2.5-flash results/l2l_qwen3-8b.Chat.En.json
```


import os
import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig,GenerationConfig
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.01,model_format="awq")

# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-1_8b-4bit'
os.system(f'git clone https://code.openxlab.org.cn/q4171119/lesson-lmdeploy.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

pipe = pipeline(base_path,
                backend_config=backend_config)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
       
def chat(message,history):
    response = pipe(message,
                    gen_config = gen_config)
    return response.text

demo = gr.ChatInterface(
            fn = chat,
            title="InternLM2-Chat-1.8_4bit-self-cognition",
            description="""InternLM is mainly developed by Shanghai AI Laboratory.  """,
            )
demo.queue(1).launch()

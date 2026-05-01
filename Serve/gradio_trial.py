#这段代码使用了 Python 的 gradio 库来创建一个简单的 Web 应用程序，允许用户输入文本并获取一个问候语作为响应.用于演示如何使用 Gradio 库快速创建一个交互式的 Web 应用程序。
import gradio as gr
def greet(name):
    return "Hello " + name + "!"
demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(share=True)
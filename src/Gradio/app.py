import gradio as gr
import numpy as np
from scipy import signal
import os
import pathlib

def getPath(path):
    return os.path.join(pathlib.Path(__file__).parent.resolve(), path)

def greet(name):
    return "Hello" + name +"!"

array = np.zeros((7, 7))

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.array(gray)

def runConvolution(inputs, matrix, matrix2, times):
    times = int(times)
    conv = np.array(matrix).astype(float)
    conv2  = np.array(matrix2).astype(float)
    
    if len(inputs.shape) == 3:
        inputs = rgb2gray(inputs)

    out = inputs / 256

    for x in range(times):
        out = signal.convolve2d(out, conv, mode="same")
        out = signal.convolve2d(out, conv2, mode="same")

    out = np.clip(out, a_min=-1, a_max = 1)
    
    return out

#demo = gr.Interface(fn=greet, inputs=gr.Image(type="pil"), outputs = gr.Image(), examples=["test.jpg"])
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            inputs = gr.Image(type="numpy", label="Input Image")
            gr.Examples([getPath("cat.jpg")], inputs=[inputs])
            matrix = gr.DataFrame(col_count=3, row_count=3, interactive=True, label="Convolution", type="numpy", datatype="number")
            gr.Examples([getPath("matrix1.csv")], inputs=[matrix])
            matrix2 = gr.DataFrame(col_count=3, row_count=3, interactive=True, label="Convolution 2", type="numpy", datatype="number")
            gr.Examples([getPath("matrixID.csv")], inputs=[matrix2])
        with gr.Column():
            outputs = gr.Image(type="pil", label="Output Image")
            btn = gr.Button("Run x times")
            times = gr.Text("10", label="x")
            btn.click(fn=runConvolution, inputs=[inputs, matrix, matrix2, times], outputs=[outputs])
        


#    inputs = gr.Image(type="pil")
    #inputs = gr.Interface(fn=greet, inputs=gr.Image(type="pil"), outputs = gr.Image(), examples=["test.jpg"])
    #matrix = gr.DataFrame(col_count=3, row_count=3, interactive=True, label="Convolution")
#    examples= gr.Examples([])
#demo = gr.Parallel(demo, exp)

#with gr.Blocks() as demo:
#    inputs = gr.Image(type="pil")
#    matrix = gr.DataFrame(col_count=3, row_count=3, interactive=True, label="Convolution")
#    examples= gr.Examples([])

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()